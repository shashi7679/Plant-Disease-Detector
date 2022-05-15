import torch.nn as nn
from torchvision.datasets import resnet18, resnet50
import config

def get_model():
    if config.MODEL=='ResNet18':
        model = resnet18(pretrained=True)
        num_fltrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_fltrs, config.N_CLASSES),
            nn.LogSoftmax(dim=1)
        )
        return model
    
    elif config.MODEL=='ResNet50':
        model = resnet50(pretrained=True)
        num_fltrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_fltrs, config.N_CLASSES),
            nn.LogSoftmax(dim=1)
        )
        return model