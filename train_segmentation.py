import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp

from data_loader import SegmentationDataset
from augmentation import get_segmentation_train_transforms, get_segmentation_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def compute_iou(pred, target, num_classes):
    """Compute IoU for binary segmentation (tooth vs background)"""
    # pred: [B, H, W] with class predictions
    # target: [B, H, W] with 0=background, 1=tooth
    
    pred_tooth = (pred > 0).float()
    target_tooth = (target > 0).float()
    
    intersection = (pred_tooth * target_tooth).sum()
    union = pred_tooth.sum() + target_tooth.sum() - intersection
    
    if union == 0:
        return 0.0
    
    iou = (intersection / union).item()
    return iou

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Compute IoU
        preds = outputs.argmax(dim=1)
        iou = compute_iou(preds, masks, num_classes)
        running_iou += iou
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_iou

def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Compute IoU
            preds = outputs.argmax(dim=1)
            iou = compute_iou(preds, masks, num_classes)
            running_iou += iou
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_iou

class CombinedLoss(nn.Module):
    """Dice Loss + BCE for binary segmentation"""
    def __init__(self, num_classes):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
    
    def forward(self, pred, target):
        # For binary: pred shape [B, 2, H, W], take channel 1 (tooth)
        pred_tooth = pred[:, 1, :, :]
        target_binary = (target > 0).float()
        return 0.7 * self.dice(pred_tooth, target_binary) + 0.3 * self.bce(pred_tooth, target_binary)

def main():
    print("\n" + "="*60)
    print("TRAINING SEGMENTATION MODEL (U-Net + ResNet50)")
    print("="*60)
    
    # Load config
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model3_segmentation']['data_dir']
    model_save_path = base_dir / config['model3_segmentation']['model_save_path']
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = base_dir / config['model3_segmentation']['meta_json']
    
    num_classes = config['model3_segmentation']['num_classes']
    input_size = tuple(config['model3_segmentation']['input_size'])
    batch_size = config['model3_segmentation']['batch_size']
    epochs = config['model3_segmentation']['epochs']
    lr = config['model3_segmentation']['lr']
    random_seed = config['project']['random_seed']
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Validate data directory
    train_img_dir = data_dir / "train" / "img"
    train_ann_dir = data_dir / "train" / "ann"
    val_img_dir = data_dir / "val" / "img"
    val_ann_dir = data_dir / "val" / "ann"
    
    if not train_img_dir.exists():
        print(f"\n❌ Error: Training data not found at {train_img_dir}")
        print("Please run: python prepare_data.py")
        return
    
    print(f"\n✓ Data directory validated: {data_dir}")
    
    # Create datasets
    train_transform = get_segmentation_train_transforms(input_size)
    val_transform = get_segmentation_val_transforms(input_size)
    
    train_dataset = SegmentationDataset(train_img_dir, train_ann_dir, meta_json, transform=train_transform)
    val_dataset = SegmentationDataset(val_img_dir, val_ann_dir, meta_json, transform=val_transform)
    
    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Task: Binary segmentation (tooth vs background)")
    print(f"✓ Number of classes: {num_classes + 1} (background + tooth)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model - Use EfficientNet encoder (better than ResNet50)
    print(f"\n✓ Creating U-Net with EfficientNet-B3 encoder (pretrained on ImageNet)")
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes + 1  # +1 for background class
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(num_classes + 1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training history
    history = []
    best_val_iou = 0.0
    patience = 10
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_iou = validate(model, val_loader, criterion, device, num_classes)
        
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_iou': val_iou
        })
        
        scheduler.step()
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Model saved (best val IoU: {best_val_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    print(f"\n✓ Training completed at epoch {epoch + 1}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = model_save_path.parent / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n✓ Training history saved to {history_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {model_save_path}")
    
    if best_val_iou > 0.70:
        print("✓ Target achieved (mean IoU > 0.70)")
    else:
        print("⚠️  Below target (mean IoU should be > 0.70)")
    
    print("\nNext step: Run python eval_segmentation.py")

if __name__ == "__main__":
    main()
