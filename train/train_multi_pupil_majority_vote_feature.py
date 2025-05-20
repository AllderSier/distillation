import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..data.datasets import get_cifar10_loaders
from ..student.student_model import get_student_model
from ..teachers.teacher_custom import get_teacher_custom

class StudentModelVariant(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModelVariant, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class StudentModelAlternative(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModelAlternative, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(96 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    model.train()
    return 100.0 * correct / total

def train_student_feature_based(
    student_model,
    feat_proj,
    teacher_model,
    train_loader,
    test_loader,
    device,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    temperature=4.0,
    epochs=5,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    log_file=None,
    teacher_path=None
):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    criterion_feat = nn.MSELoss()

    optimizer = optim.SGD(
        list(student_model.parameters()) + list(feat_proj.parameters()),
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    teacher_model.eval()

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Teacher path:", teacher_path])
            writer.writerow(["# alpha:", alpha, "# beta:", beta, "# gamma:", gamma, "# temperature:", temperature])
            writer.writerow(["# lr:", lr, "# epochs:", epochs])
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits_student = student_model(images)
            features_student = student_model.features(images)

            with torch.no_grad():
                logits_teacher = teacher_model(images)
                features_teacher = teacher_model.features(images)

            loss_ce = criterion_ce(logits_student, labels)

            student_logits_T = F.log_softmax(logits_student / temperature, dim=1)
            teacher_soft = F.softmax(logits_teacher / temperature, dim=1)
            loss_kd = criterion_kd(student_logits_T, teacher_soft) * (temperature ** 2)

            fs = feat_proj(features_student)
            if fs.shape[-2:] != features_teacher.shape[-2:]:
                fs = nn.AdaptiveAvgPool2d(features_teacher.shape[-2:])(fs)
            loss_feat = criterion_feat(fs, features_teacher)

            loss = alpha * loss_ce + beta * loss_kd + gamma * loss_feat
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        test_acc = evaluate(student_model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if log_file is not None:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{test_acc:.2f}"])

    return student_model

def majority_vote(predictions):
    stacked = torch.stack(predictions, dim=0)
    mode, _ = torch.mode(stacked, dim=0)
    return mode

def ensemble_evaluate(student_models, data_loader, device):
    total = 0
    correct = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        preds = []
        for model in student_models:
            model.eval()
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            preds.append(pred)
        final_pred = majority_vote(preds)
        total += labels.size(0)
        correct += (final_pred == labels).sum().item()
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    teacher_path = "teacher_custom.pth"
    teacher_model = get_teacher_custom(num_classes=10).to(device)
    if not os.path.exists(teacher_path):
        print(f"Teacher weights not found: '{teacher_path}'. Please train the teacher first!")
        return
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    print("Teacher loaded.")

    student_models = []

    student1 = get_student_model(num_classes=10).to(device)
    feat_proj1 = nn.Conv2d(128, 512, kernel_size=1).to(device)
    student1 = train_student_feature_based(
        student_model=student1,
        feat_proj=feat_proj1,
        teacher_model=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        alpha=1.0, beta=1.0, gamma=1.0,
        temperature=4.0, epochs=15,
        lr=0.01, momentum=0.9, weight_decay=5e-4,
        log_file="student1_feature_kd_ce_log.csv",
        teacher_path=teacher_path
    )
    torch.save(student1.state_dict(), "student1_feature_kd_ce.pth")
    student_models.append(student1)
    print("Student 1 trained and saved.")

    student2 = StudentModelVariant(num_classes=10).to(device)
    feat_proj2 = nn.Conv2d(64, 512, kernel_size=1).to(device)
    student2 = train_student_feature_based(
        student_model=student2,
        feat_proj=feat_proj2,
        teacher_model=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        alpha=1.0, beta=1.0, gamma=1.0,
        temperature=4.0, epochs=15,
        lr=0.01, momentum=0.9, weight_decay=5e-4,
        log_file="student2_feature_kd_ce_log.csv",
        teacher_path=teacher_path
    )
    torch.save(student2.state_dict(), "student2_feature_kd_ce.pth")
    student_models.append(student2)
    print("Student 2 trained and saved.")

    student3 = StudentModelAlternative(num_classes=10).to(device)
    feat_proj3 = nn.Conv2d(96, 512, kernel_size=1).to(device)
    student3 = train_student_feature_based(
        student_model=student3,
        feat_proj=feat_proj3,
        teacher_model=teacher_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        alpha=1.0, beta=1.0, gamma=1.0,
        temperature=4.0, epochs=15,
        lr=0.01, momentum=0.9, weight_decay=5e-4,
        log_file="student3_feature_kd_ce_log.csv",
        teacher_path=teacher_path
    )
    torch.save(student3.state_dict(), "student3_feature_kd_ce.pth")
    student_models.append(student3)
    print("Student 3 trained and saved.")

    ensemble_acc = ensemble_evaluate(student_models, test_loader, device)
    print(f"Ensemble Accuracy (feature-based majority vote): {ensemble_acc:.2f}%")

if __name__ == "__main__":
    main()
