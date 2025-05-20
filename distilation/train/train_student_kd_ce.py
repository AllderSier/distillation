import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distilation.data.datasets import get_cifar10_loaders
from distilation.student.student_model import get_student_model

from distilation.teachers.teacher_custom import get_teacher_custom
from distilation.teachers.teacher_resnet18 import get_teacher_resnet18
from distilation.teachers.teacher_vgg16 import get_teacher_vgg16


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


def train_student_kd(student_model, teacher_model,
                     train_loader, test_loader, device,
                     alpha=0.5, temperature=4.0, epochs=15,
                     lr=0.01, momentum=0.9, weight_decay=5e-4,
                     log_file=None, teacher_path=None):
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.SGD(student_model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    teacher_model.eval()

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Teacher path:", teacher_path])
            writer.writerow(["# alpha:", alpha, "# temperature:", temperature])
            writer.writerow(["# lr:", lr, "# epochs:", epochs])
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs_student = student_model(images)

            with torch.no_grad():
                outputs_teacher = teacher_model(images)

            loss_ce = criterion_ce(outputs_student, labels)

            student_logits_T = F.log_softmax(outputs_student / temperature, dim=1)
            teacher_logits_T = F.softmax(outputs_teacher / temperature, dim=1)
            loss_kd = criterion_kd(student_logits_T, teacher_logits_T) * (temperature**2)

            loss = alpha * loss_ce + (1 - alpha) * loss_kd
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

    return student_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    path_custom = 'teacher_custom.pth'
    path_resnet = 'teacher_resnet16.pth'
    path_vgg    = 'teacher_vgg16.pth'

    log_custom_student  = 'student_kd_ce_log_custom.csv'
    log_resnet_student  = 'student_kd_ce_log_resnet.csv'
    log_vgg_student     = 'student_kd_ce_log_vgg.csv'

    teachers_info = [
        ("custom",   get_teacher_custom(num_classes=10).to(device),   path_custom, log_custom_student),
        ("resnet16", get_teacher_resnet18(num_classes=10).to(device), path_resnet, log_resnet_student),
        ("vgg16",    get_teacher_vgg16(num_classes=10, pretrained=False).to(device), path_vgg, log_vgg_student),
    ]

    alpha = 0.5
    temperature = 4.0
    epochs = 15
    lr = 0.01

    for teacher_name, teacher_model, teacher_path, student_log in teachers_info:
        if not os.path.exists(teacher_path):
            print(f"\n[WARNING] Teacher weights not found: '{teacher_path}'. "
                  f"Для {teacher_name} нужно сначала обучить учителя.")
            continue

        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"\nTeacher ({teacher_name}) loaded from '{teacher_path}'.")

        student_model = get_student_model(num_classes=10).to(device)

        student_weights_path = f"student_kd_ce_{teacher_name}.pth"

        student_model = train_student_kd(
            student_model=student_model,
            teacher_model=teacher_model,
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
            teacher_path=teacher_path
        )

        torch.save(student_model.state_dict(), student_weights_path)
        print(f"\n[Done] Student (KD) for teacher '{teacher_name}' saved to '{student_weights_path}'. "
              f"Log file: '{student_log}'.")


if __name__ == "__main__":
    main()
