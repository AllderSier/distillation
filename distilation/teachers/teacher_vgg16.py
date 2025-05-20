import torch.nn as nn
from torchvision.models import vgg16

def get_teacher_vgg16(num_classes=10, pretrained=False):

    model = vgg16(pretrained=pretrained)
    
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model