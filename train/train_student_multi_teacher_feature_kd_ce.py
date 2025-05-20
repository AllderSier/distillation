# train/train_student_multi_teacher_feature_kd_ce.py

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
        raise ValueError("Unknown architecture")

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

def train_student_multi_teacher_feature_kd_ce(
    student_model,
    feat_proj, 
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
    teacher_ids=None,  
    feature_loss_weight=1.0
):

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    criterion_feature = nn.MSELoss()

    optimizer = optim.SGD(list(student_model.parameters()) + list(feat_proj.parameters()),
                          lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for teacher in teacher_models:
        teacher.eval()

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Teacher IDs:", teacher_ids])
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

            teacher_logits_list = []
            teacher_feats_list = []
            with torch.no_grad():
                for teacher in teacher_models:
                    logits_t = teacher(images)
                    teacher_logits_list.append(logits_t)

                    t_feats = extract_teacher_features(teacher, images)
                    target_spatial = student_feats.shape[-2:] 
                    t_feats = nn.AdaptiveAvgPool2d(target_spatial)(t_feats)
                    teacher_feats_list.append(t_feats)
            
            agg_teacher_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)
            agg_teacher_feats = torch.mean(torch.stack(teacher_feats_list, dim=0), dim=0)

            student_logits_T = F.log_softmax(logits_student / temperature, dim=1)
            teacher_soft = F.softmax(agg_teacher_logits / temperature, dim=1)
            loss_kd = criterion_kd(student_logits_T, teacher_soft) * (temperature ** 2)


            if projected_student_feats.shape[-1] != agg_teacher_feats.shape[-1]:
                projected_student_feats_aligned = nn.AdaptiveAvgPool2d(agg_teacher_feats.shape[-2:])(projected_student_feats)
            else:
                projected_student_feats_aligned = projected_student_feats

            loss_feature = criterion_feature(projected_student_feats_aligned, agg_teacher_feats)

            total_loss = alpha * loss_ce + (1 - alpha) * (loss_kd + feature_loss_weight * loss_feature)
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

    return student_model, feat_proj

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
        teacher_models.append(t_model)

    if not teacher_models:
        print("No teacher.")
        return

    student_model = get_student_model(num_classes=10).to(device)
    feat_proj = nn.Conv2d(128, 512, kernel_size=1).to(device)

    log_file = "student_multi_teacher_feature_kd_ce_log.csv"
    student_weights_path = "student_multi_teacher_feature_kd_ce.pth"

    alpha = 0.5
    temperature = 4.0
    epochs = 15
    lr = 0.01
    feature_loss_weight = 1.0

    student_model, feat_proj = train_student_multi_teacher_feature_kd_ce(
        student_model=student_model,
        feat_proj=feat_proj,
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
        teacher_ids=teacher_ids,
        feature_loss_weight=feature_loss_weight
    )

    torch.save(student_model.state_dict(), student_weights_path)
    print(f"\n[Done] Student (multi-teacher feature-based KD с CE) saved as '{student_weights_path}'. Log: '{log_file}'.")

if __name__ == "__main__":
    main()
