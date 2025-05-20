import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..teachers.teacher_custom import get_teacher_custom
from ..student.student_model import get_student_model
from ..data.datasets import get_cifar10_loaders

class RandomGenerator(nn.Module):
    def __init__(self, image_shape=(3, 32, 32)):
        super(RandomGenerator, self).__init__()
        self.image_shape = image_shape

    def forward(self, batch_size, device):
        return torch.rand(batch_size, *self.image_shape, device=device)

def train_student_data_free_random(
    teacher_model,
    student_model,
    generator,
    device,
    train_loader,
    test_loader,
    *,
    epochs           = 15,
    steps_per_epoch  = 100,
    batch_size       = 64,
    use_soft_targets = False,
    temperature      = 4.0,
    lr               = 0.01,
    log_file         = "student_data_free_random_log.csv",
):
    csv_fh = None
    csv_writer = None
    if log_file:
        csv_fh = open(log_file, mode="w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch", "avg_loss", "train_acc", "test_acc"])

    optimizer   = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')

    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    student_model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            synth = generator(batch_size, device)
            with torch.no_grad():
                logits_teacher = teacher_model(synth)

            optimizer.zero_grad()
            logits_student = student_model(synth)

            if use_soft_targets:
                student_soft = F.log_softmax(logits_student / temperature, dim=1)
                teacher_soft = F.softmax(logits_teacher / temperature, dim=1)
                loss = criterion_kd(student_soft, teacher_soft) * (temperature ** 2)
            else:
                _, labels = torch.max(logits_teacher, dim=1)
                loss = criterion_ce(logits_student, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / steps_per_epoch

        student_model.eval()
        correct_train = total_train = 0
        with torch.no_grad():
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                preds = student_model(images).argmax(dim=1)
                correct_train += (preds == targets).sum().item()
                total_train += targets.size(0)
        train_acc = 100.0 * correct_train / total_train

        correct_test = total_test = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                preds = student_model(images).argmax(dim=1)
                correct_test += (preds == targets).sum().item()
                total_test += targets.size(0)
        test_acc = 100.0 * correct_test / total_test

        print(f"Epoch [{epoch}/{epochs}] Avg Loss: {avg_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% Test Acc: {test_acc:.2f}%")

        if csv_writer:
            csv_writer.writerow([
                epoch,
                f"{avg_loss:.6f}",
                f"{train_acc:.2f}",
                f"{test_acc:.2f}"
            ])
            csv_fh.flush()
        student_model.train()

    if csv_fh:
        csv_fh.close()

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

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    teacher_path = "teacher_custom.pth"
    teacher = get_teacher_custom(num_classes=10).to(device)
    if not os.path.exists(teacher_path):
        print(f"Weights not found: '{teacher_path}'. Please train the teacher first.")
        return
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    print("Teacher loaded.")

    student = get_student_model(num_classes=10).to(device)
    gen = RandomGenerator(image_shape=(3, 32, 32))

    student = train_student_data_free_random(
        teacher_model   = teacher,
        student_model   = student,
        generator       = gen,
        device          = device,
        train_loader    = train_loader,
        test_loader     = test_loader,
        epochs          = 15,
        steps_per_epoch = 100,
        batch_size      = 64,
        use_soft_targets= True,
        temperature     = 4.0,
        lr              = 0.01,
        log_file        = "student_data_free_random_log.csv"
    )

    torch.save(student.state_dict(), "student_data_free_random.pth")
    print("Training complete. Weights saved to 'student_data_free_random.pth', log → 'student_data_free_random_log.csv'")

if __name__ == "__main__":
    main()
