import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_classification_train_transforms(input_size):
    """Augmentation for classification training"""
    return transforms.Compose([
        transforms.Resize((int(input_size[0] * 1.1), int(input_size[1] * 1.1))),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])

def get_classification_val_transforms(input_size):
    """No augmentation for classification validation/test"""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_segmentation_train_transforms(input_size):
    """Augmentation for segmentation training (applies to both image and mask)"""
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_segmentation_val_transforms(input_size):
    """No augmentation for segmentation validation/test"""
    return A.Compose([
        A.Resize(height=input_size[0], width=input_size[1]),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
