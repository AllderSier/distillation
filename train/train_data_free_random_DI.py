import os
import csv
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..teachers.teacher_custom import get_teacher_custom
from ..student.student_model import get_student_model
from ..data.datasets import get_cifar10_loaders

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
    return tv_h + tv_w

class DeepInversionSynthesizer:
    def __init__(
        self,
        teacher: nn.Module,
        image_shape=(3, 32, 32),
        iters: int = 50,
        lr: float = 0.1,
        bn_coeff: float = 0.2,
        tv_coeff: float = 1e-4,
        ent_coeff: float = 0.0,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.CHW = image_shape
        self.iters = iters
        self.lr = lr
        self.bn_coeff = bn_coeff
        self.tv_coeff = tv_coeff
        self.ent_coeff = ent_coeff
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._bn_loss: torch.Tensor | None = None
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                h = m.register_forward_hook(self._bn_hook)
                self._handles.append(h)

    def _bn_hook(self, module: nn.BatchNorm2d, inputs, outputs):
        x = inputs[0]
        μ_a = x.mean((0, 2, 3))
        σ2_a = x.var((0, 2, 3), unbiased=False)
        loss = F.mse_loss(μ_a, module.running_mean, reduction="sum") + \
               F.mse_loss(σ2_a, module.running_var, reduction="sum")
        self._bn_loss = loss if self._bn_loss is None else self._bn_loss + loss

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        with torch.enable_grad():
            images = torch.randn(batch_size, *self.CHW, device=device, requires_grad=True)
            opt = optim.Adam([images], lr=self.lr, betas=(0.5, 0.9))
            for _ in range(self.iters):
                opt.zero_grad()
                self._bn_loss = None
                logits_t = self.teacher(images)
                loss_bn = self._bn_loss if self._bn_loss is not None else torch.zeros(1, device=device)
                loss_tv = total_variation(images)
                probs = F.softmax(logits_t, dim=1)
                loss_ent = (probs * torch.log(probs + 1e-6)).sum(1).mean()
                loss = self.bn_coeff * loss_bn + self.tv_coeff * loss_tv + self.ent_coeff * loss_ent
                loss.backward()
                opt.step()
                images.data.clamp_(0, 1)
        return images.detach()

    def __del__(self):
        for h in self._handles:
            h.remove()

def train_student_deepinv_kd(
    teacher_model: nn.Module,
    student_model: nn.Module,
    synthesizer: DeepInversionSynthesizer,
    device: torch.device,
    train_loader,
    test_loader,
    *,
    epochs: int = 300,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
    ce_weight: float = 0.05,
    temperature: float = 2.0,
    lr: float = 0.01,
    log_file: str | None = "student_deepinv_kd_log.csv",
):
    kd_weight = 1.0 - ce_weight
    csv_fh = csv_writer = None
    if log_file:
        csv_fh = open(log_file, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["epoch", "kd", "ce", "train_acc", "test_acc"])
    opt_s = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    student_model.train()
    for epoch in range(1, epochs + 1):
        kd_sum = ce_sum = 0.0
        for _ in range(steps_per_epoch):
            imgs = synthesizer(batch_size, device)
            with torch.no_grad():
                log_t = teacher_model(imgs)
            log_s = student_model(imgs)
            kd = kd_loss_fn(F.log_softmax(log_s / temperature, 1), F.softmax(log_t / temperature, 1)) * temperature**2
            ce = ce_loss_fn(log_s, log_t.argmax(1))
            loss = kd_weight * kd + ce_weight * ce
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()
            kd_sum += kd.item()
            ce_sum += ce.item()
        def accuracy(loader):
            corr = tot = 0
            student_model.eval()
            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    corr += (student_model(x).argmax(1) == y).sum().item()
                    tot += y.size(0)
            student_model.train()
            return 100.0 * corr / tot
        tr_acc = accuracy(train_loader)
        te_acc = accuracy(test_loader)
        print(f"[Epoch {epoch}/{epochs}] KD: {kd_sum/steps_per_epoch:.4f} CE: {ce_sum/steps_per_epoch:.4f} Train: {tr_acc:.2f}% Test: {te_acc:.2f}%")
        if csv_writer:
            csv_writer.writerow([
                epoch,
                f"{kd_sum/steps_per_epoch:.6f}",
                f"{ce_sum/steps_per_epoch:.6f}",
                f"{tr_acc:.2f}",
                f"{te_acc:.2f}"
            ])
            csv_fh.flush()
    if csv_fh:
        csv_fh.close()
    return student_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    teacher_path = "teacher_custom.pth"
    teacher = get_teacher_custom(10).to(device)
    if not os.path.exists(teacher_path):
        print(f"Teacher weights not found: '{teacher_path}'. Please train the teacher first.")
        return
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    print("Teacher loaded.")
    student = get_student_model(10).to(device)
    synthesizer = DeepInversionSynthesizer(
        teacher,
        image_shape=(3, 32, 32),
        iters=50,
        lr=0.1,
        bn_coeff=0.2,
        tv_coeff=1e-4,
        ent_coeff=0.0,
    )
    student = train_student_deepinv_kd(
        teacher_model=teacher,
        student_model=student,
        synthesizer=synthesizer,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=15,
        steps_per_epoch=100,
        batch_size=64,
        ce_weight=0.05,
        temperature=2.0,
        lr=0.01,
        log_file="student_deepinv_kd_log.csv",
    )
    torch.save(student.state_dict(), "student_deepinv_kd.pth")
    print("Training complete. Weights saved to 'student_deepinv_kd.pth'. Log → 'student_deepinv_kd_log.csv'")

if __name__ == "__main__":
    main()
