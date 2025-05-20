import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            # Блок 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Блок 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Блок 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)              
        x = x.view(x.size(0), -1)     
        x = self.classifier(x)          
        return x


def get_student_model(num_classes=10):
    return StudentModel(num_classes=num_classes)
