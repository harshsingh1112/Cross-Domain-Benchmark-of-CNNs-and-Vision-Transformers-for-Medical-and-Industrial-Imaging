import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
import os
import ssl

# Fix for SSL certificate errors during download
ssl._create_default_https_context = ssl._create_unverified_context

def get_eurosat_loaders(root='./data', batch_size=32, num_workers=4, download=True):
    """
    Returns data loaders for EuroSAT. 
    Since EuroSAT usually doesn't have standard splits in torchvision, we will create random splits.
    """
    
    # ImageNet normalization statistics
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if download is needed or data exists
    if download:
        # Note: torchvision EuroSAT might need valid SSL context or manual download if URL is flaky.
        try:
           full_dataset = EuroSAT(root=root, transform=data_transform, download=True)
        except Exception as e:
           print(f"Error downloading EuroSAT via torchvision: {e}")
           print("Please manually download EuroSAT if this fails.")
           raise e
    else:
        full_dataset = EuroSAT(root=root, transform=data_transform, download=False)

    # Create 80/10/10 splits
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for split reproducibility
    )
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
