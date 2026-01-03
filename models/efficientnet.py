import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes):
    """
    Returns an EfficientNet-B0 model pretrained on ImageNet, adapted for num_classes.
    """
    print("Loading EfficientNet-B0...")
    # Use weights='IMAGENET1K_V1' (or 'DEFAULT') per torchvision >= 0.13
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    # Replace the final fully connected layer
    # EfficientNet has a 'classifier' sequential block
    # Check structure: usually model.classifier[1] is the Linear layer
    
    # Re-create the classifier block to be safe or just replace the last layer
    num_ftrs = model.classifier[1].in_features
    
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model
