import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import PathMNIST

def get_pathmnist_loaders(batch_size=32, num_workers=4, download=True):
    """
    Returns data loaders for PathMNIST.
    """
    
    # ImageNet normalization statistics
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Train Data
    train_dataset = PathMNIST(split='train', transform=data_transform, download=download, size=224)
    
    # Validation Data (Using 'val' split)
    val_dataset = PathMNIST(split='val', transform=data_transform, download=download, size=224)

    # Test Data
    test_dataset = PathMNIST(split='test', transform=data_transform, download=download, size=224)
    
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
