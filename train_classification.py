import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from data_loader import ClassificationDataset
from augmentation import get_classification_train_transforms, get_classification_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL (ResNet18)")
    print("="*60)
    
    # Load config
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model1_classification']['data_dir']
    model_save_path = base_dir / config['model1_classification']['model_save_path']
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_classes = config['model1_classification']['num_classes']
    input_size = tuple(config['model1_classification']['input_size'])
    batch_size = config['model1_classification']['batch_size']
    phase1_epochs = config['model1_classification']['phase1_epochs']
    phase2_epochs = config['model1_classification']['phase2_epochs']
    phase1_lr = config['model1_classification']['phase1_lr']
    phase2_lr = config['model1_classification']['phase2_lr']
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
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    if not train_dir.exists():
        print(f"\n❌ Error: Training data not found at {train_dir}")
        print("Please run: python prepare_data.py")
        return
    
    print(f"\n✓ Data directory validated: {data_dir}")
    
    # Create datasets
    train_transform = get_classification_train_transforms(input_size)
    val_transform = get_classification_val_transforms(input_size)
    
    train_dataset = ClassificationDataset(train_dir, transform=train_transform)
    val_dataset = ClassificationDataset(val_dir, transform=val_transform)
    
    print(f"\n✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Classes: {train_dataset.class_names}")
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=train_dataset.labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n✓ Class weights (for imbalance handling):")
    for i, (cls_name, weight) in enumerate(zip(train_dataset.class_names, class_weights)):
        print(f"   {cls_name}: {weight:.3f}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n✓ Loading ResNet18 (pretrained on ImageNet)")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=config['model1_classification']['dropout']),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training history
    history = []
    best_val_acc = 0.0
    
    # ========== PHASE 1: Train only classifier head ==========
    print("\n" + "="*60)
    print("PHASE 1: Training classifier head (backbone frozen)")
    print("="*60)
    
    # Freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(model.fc.parameters(), lr=phase1_lr)
    
    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch+1}/{phase1_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        history.append({
            'epoch': epoch + 1,
            'phase': 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Model saved (best val acc: {best_val_acc:.2f}%)")
    
    # ========== PHASE 2: Fine-tune top layers ==========
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning layer4 + fc (lower learning rate)")
    print("="*60)
    
    # Unfreeze layer4 (last residual block)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=phase2_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(phase2_epochs):
        print(f"\nEpoch {phase1_epochs + epoch + 1}/{phase1_epochs + phase2_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        history.append({
            'epoch': phase1_epochs + epoch + 1,
            'phase': 2,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ Model saved (best val acc: {best_val_acc:.2f}%)")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✓ Early stopping triggered (no improvement for {patience} epochs)")
                break
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = model_save_path.parent / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\n✓ Training history saved to {history_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_save_path}")
    print("\nNext step: Run python eval_classification.py")

if __name__ == "__main__":
    main()
