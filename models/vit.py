import timm
import torch.nn as nn

def get_vit(num_classes):
    """
    Returns a Vision Transformer (ViT-Tiny) pretrained on ImageNet, adapted for num_classes.
    """
    print("Loading ViT-Tiny (timms)...")
    # ViT-Tiny: 'vit_tiny_patch16_224'
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    
    return model
