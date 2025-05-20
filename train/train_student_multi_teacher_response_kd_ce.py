# train/train_student_multi_teacher_response_kd_ce.py

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

def train_student_multi_teacher_response_kd_ce(
    student_model,
    teacher_models, 
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
    teacher_ids=None 
):
   
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    optimizer = optim.SGD(student_model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for teacher in teacher_models:
        teacher.eval()

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Teacher IDs:", teacher_ids])
            writer.writerow(["# alpha:", alpha, "# temperature:", temperature])
            writer.writerow(["# lr:", lr, "# epochs:", epochs])
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits_student = student_model(images)
            loss_ce = criterion_ce(logits_student, labels)

            teacher_logits_list = []
            with torch.no_grad():
                for teacher in teacher_models:
                    teacher_logits = teacher(images)
                    teacher_logits_list.append(teacher_logits)
            
            agg_teacher_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)

            student_logits_T = F.log_softmax(logits_student / temperature, dim=1)
            teacher_soft = F.softmax(agg_teacher_logits / temperature, dim=1)
            loss_kd = criterion_kd(student_logits_T, teacher_soft) * (temperature ** 2)

            total_loss = alpha * loss_ce + (1 - alpha) * loss_kd
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

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

    teacher_ids = ["custom", "resnet18", "vgg16"]
    teacher_custom = get_teacher_custom(num_classes=10).to(device)
    teacher_resnet = get_teacher_resnet18(num_classes=10).to(device)
    teacher_vgg = get_teacher_vgg16(num_classes=10, pretrained=False).to(device)
    
    teacher_paths = {
        "custom": "teacher_custom.pth",
        "resnet18": "teacher_resnet16.pth",
        "vgg16": "teacher_vgg16.pth"
    }
    
    teachers_info = [
        (teacher_ids[0], teacher_custom, teacher_paths["custom"]),
        (teacher_ids[1], teacher_resnet, teacher_paths["resnet18"]),
        (teacher_ids[2], teacher_vgg, teacher_paths["vgg16"])
    ]
    
    teacher_models = []
    for tid, t_model, t_path in teachers_info:
        if not os.path.exists(t_path):
            continue
        t_model.load_state_dict(torch.load(t_path, map_location=device))
        print(f"\nTeacher ({tid}) upload as '{t_path}'.")
        teacher_models.append(t_model)
    
    if not teacher_models:
        print("No teacher.")
        return

    student_model = get_student_model(num_classes=10).to(device)
    
    log_file = "student_multi_teacher_response_kd_ce_log.csv"
    student_weights_path = "student_multi_teacher_response_kd_ce.pth"
    
    alpha = 0.5
    temperature = 4.0
    epochs = 15
    lr = 0.01

    student_model = train_student_multi_teacher_response_kd_ce(
        student_model=student_model,
        teacher_models=teacher_models,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        alpha=alpha,
        temperature=temperature,
        epochs=epochs,
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        log_file=log_file,
        teacher_ids=teacher_ids
    )

    torch.save(student_model.state_dict(), student_weights_path)
    print(f"\n[Done] Student (multi-teacher response-based KD с CE) saved as '{student_weights_path}'. Log: '{log_file}'.")

if __name__ == "__main__":
    main()
