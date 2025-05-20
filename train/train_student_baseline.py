import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from distilation.data.datasets import get_cifar10_loaders
from distilation.student.student_model import get_student_model

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


def train_student_baseline(
    student_model,
    train_loader,
    test_loader,
    device,
    epochs=15,
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    log_file=None
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student_model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if log_file is not None:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["# Baseline training, no teacher"])
            writer.writerow(["# lr:", lr, "# epochs:", epochs])
            writer.writerow(["epoch", "loss", "test_acc"])

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = student_model(images)
            loss = criterion(outputs, labels)
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
    student_model = get_student_model(num_classes=10).to(device)

    epochs = 15
    lr = 0.01
    student_weights_path = "student_baseline.pth"
    student_log_path = "student_baseline_log.csv"

    student_model = train_student_baseline(
        student_model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        log_file=student_log_path
    )

    torch.save(student_model.state_dict(), student_weights_path)
    print(f"\nBaseline Student saved as '{student_weights_path}'. Log: '{student_log_path}'.")    


if __name__ == "__main__":
    main()
