import torch
import torch.nn as nn
from torchvision import models

def get_resnet(num_classes):
    """
    Returns a ResNet-50 model pretrained on ImageNet, adapted for num_classes.
    """
    print("Loading ResNet-50...")
    # Use weights='IMAGENET1K_V1' (or 'DEFAULT') for ImageNet pretrained weights
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
