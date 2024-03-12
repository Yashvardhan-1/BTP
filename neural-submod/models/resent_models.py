import torch.nn as nn
import torchvision

def get_resent18_model(num_classes=10):
    model = torchvision.models.resnet18(weights=None)  # Use 'weights' for pretrained models
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_resent101_model(num_classes=10):
    model = torchvision.models.resnet101(weights=None)  # Use 'weights' for pretrained models
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model