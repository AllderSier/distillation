import torch.nn as nn
import torchvision.models as models

def get_teacher_resnet18(num_classes=10, pretrained=False):

    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model