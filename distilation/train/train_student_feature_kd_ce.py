# train/train_student_feature_kd_ce.py

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..data.datasets import get_cifar10_loaders
from ..student.student_model import get_student_model
from ..teachers.teacher_custom import get_teacher_custom
from ..teachers.teacher_resnet18 import get_teacher_resnet18
from ..teachers.teacher_vgg16 import get_teacher_vgg16

def extract_teacher_features(teacher_model, images):
    if hasattr(teacher_model, 'features'):
        return teacher_model.features(images)
    elif hasattr(teacher_model, 'layer4'):
        x = teacher_model.conv1(images)
        x = teacher_model.bn1(x)
        x = teacher_model.relu(x)
        x = teacher_model.maxpool(x)
        x = teacher_model.layer1(x)
        x = teacher_model.layer2(x)
        x = teacher_model.layer3(x)
        x = teacher_model.layer4(x)
        return x
    else:
        raise ValueError("НUnknown teacher")

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100.0 * correct / total

def train_student_feature_kd_ce(
    student_model,
    teacher_model,
    feat_proj, 
    train_loader,
    test_loader,
    device,
    alpha=0.5,
    temperature=4.0,
    epochs=5,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    log_file=None,
    teacher_path=None,
    feature_loss_weight=1.0
):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_feature = nn.MSELoss()

    optimizer = optim.SGD(list(student_model.parameters()) + list(feat_proj.parameters()),
                          lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    teacher_model.eval()

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Teacher path:", teacher_path])
            writer.writerow(["# alpha:", alpha, "# temperature:", temperature, "# feature_loss_weight:", feature_loss_weight])
            writer.writerow(["# lr:", lr, "# epochs:", epochs])
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        student_model.train()
        feat_proj.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits_student = student_model(images)
            loss_ce = criterion_ce(logits_student, labels)

            student_feats = student_model.features(images)  
            projected_student_feats = feat_proj(student_feats)  

            with torch.no_grad():
                teacher_feats = extract_teacher_features(teacher_model, images)
            
            if teacher_feats.shape[-1] == 1:
                projected_student_feats_aligned = nn.AdaptiveAvgPool2d((1,1))(projected_student_feats)
            else:
                projected_student_feats_aligned = projected_student_feats

            loss_feature = criterion_feature(projected_student_feats_aligned, teacher_feats)

            loss = alpha * loss_ce + (1 - alpha) * feature_loss_weight * loss_feature
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        test_acc = evaluate(student_model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if log_file is not None:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{test_acc:.2f}"])

    return student_model, feat_proj

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    teachers_info = [
        ("custom",   get_teacher_custom(num_classes=10).to(device),   'teacher_custom.pth', 'student_feature_kd_ce_log_custom.csv'),
        ("resnet18", get_teacher_resnet18(num_classes=10).to(device), 'teacher_resnet16.pth', 'student_feature_kd_ce_log_resnet.csv'),
        ("vgg16",    get_teacher_vgg16(num_classes=10, pretrained=False).to(device), 'teacher_vgg16.pth', 'student_feature_kd_ce_log_vgg.csv'),
    ]

    alpha = 0.5
    temperature = 4.0  
    epochs = 15
    lr = 0.01
    feature_loss_weight = 1.0

    for teacher_name, teacher_model, teacher_path, student_log in teachers_info:
        if not os.path.exists(teacher_path):
            print(f"\n[WARNING] Веса учителя '{teacher_name}' не найдены: '{teacher_path}'. Обучение студента для данного учителя пропущено.")
            continue

        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"\nУчитель ({teacher_name}) загружен из '{teacher_path}'.")

        student_model = get_student_model(num_classes=10).to(device)
        feat_proj = nn.Conv2d(128, 512, kernel_size=1).to(device)

        student_weights_path = f"student_feature_kd_ce_{teacher_name}.pth"

        student_model, feat_proj = train_student_feature_kd_ce(
            student_model=student_model,
            teacher_model=teacher_model,
            feat_proj=feat_proj,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            alpha=alpha,
            temperature=temperature,
            epochs=epochs,
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
            log_file=student_log,
            teacher_path=teacher_path,
            feature_loss_weight=feature_loss_weight
        )

        torch.save(student_model.state_dict(), student_weights_path)
        print(f"\n[Done] Student (feature-based KD с CE) for teacher '{teacher_name}' saved as '{student_weights_path}'. Log: '{student_log}'.")

if __name__ == "__main__":
    main()
