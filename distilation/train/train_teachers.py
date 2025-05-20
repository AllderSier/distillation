import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim


from ..teachers.teacher_custom  import get_teacher_custom
from ..teachers.teacher_resnet18 import get_teacher_resnet18
from ..teachers.teacher_vgg16    import get_teacher_vgg16

from ..data.datasets import get_cifar10_loaders 

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return 100.0 * correct / total

def train_teacher(model, train_loader, test_loader, device,
                  lr=0.01, momentum=0.9, weight_decay=5e-4, epochs=15,
                  log_file=None):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        test_acc = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if log_file is not None:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{avg_loss:.4f}", f"{test_acc:.2f}"])

    return model

def main():
    print(torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders()

    path_custom = 'teacher_custom.pth'
    path_resnet = 'teacher_resnet16.pth'
    path_vgg    = 'teacher_vgg16.pth'

    log_custom  = 'teacher_custom_log.csv'
    log_resnet  = 'teacher_resnet16_log.csv'
    log_vgg     = 'teacher_vgg16_log.csv'

    teacher_custom = get_teacher_custom(num_classes=10).to(device)
    if os.path.exists(path_custom):
        print(f"\n[TeacherCustomModel] Found '{path_custom}'. Loading...")
        teacher_custom.load_state_dict(torch.load(path_custom, map_location=device))
    else:
        print("\n[TeacherCustomModel] Training...")
        teacher_custom = train_teacher(
            teacher_custom, train_loader, test_loader, device,
            epochs=15,
            log_file=log_custom
        )
        torch.save(teacher_custom.state_dict(), path_custom)

    teacher_resnet16 = get_teacher_resnet18(num_classes=10).to(device)
    if os.path.exists(path_resnet):
        print(f"\n[ResNet16] Found '{path_resnet}'. Loading...")
        teacher_resnet16.load_state_dict(torch.load(path_resnet, map_location=device))
    else:
        print("\n[ResNet16] Training...")
        teacher_resnet16 = train_teacher(
            teacher_resnet16, train_loader, test_loader, device,
            epochs=15,
            log_file=log_resnet
        )
        torch.save(teacher_resnet16.state_dict(), path_resnet)

    teacher_vgg16 = get_teacher_vgg16(num_classes=10, pretrained=False).to(device)
    if os.path.exists(path_vgg):
        print(f"\n[VGG16] Found '{path_vgg}'. Loading...")
        teacher_vgg16.load_state_dict(torch.load(path_vgg, map_location=device))
    else:
        print("\n[VGG16] Training...")
        teacher_vgg16 = train_teacher(
            teacher_vgg16, train_loader, test_loader, device,
            epochs=15,
            log_file=log_vgg
        )
        torch.save(teacher_vgg16.state_dict(), path_vgg)

    print("\nAll teachers are ready (loaded or newly trained).")

if __name__ == "__main__":
    main()
