import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from data_loader import ClassificationDataset
from augmentation import get_classification_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def extract_features(model, dataloader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feat = model(images)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)

def main():
    print("\n" + "="*60)
    print("SIMPLE CLASSIFICATION (Feature Extraction + LogReg)")
    print("="*60)
    
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model1_classification']['data_dir']
    model_save_path = base_dir / config['model1_classification']['model_save_path']
    model_save_path = model_save_path.parent / "simple_model.pkl"
    
    input_size = tuple(config['model1_classification']['input_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✓ Using device: {device}")
    
    # Load datasets
    transform = get_classification_val_transforms(input_size)
    train_dataset = ClassificationDataset(data_dir / "train", transform=transform)
    val_dataset = ClassificationDataset(data_dir / "val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Feature extractor (frozen ResNet18)
    print("\n✓ Loading ResNet18 feature extractor...")
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()
    
    # Extract features
    print("\n✓ Extracting training features...")
    X_train, y_train = extract_features(resnet, train_loader, device)
    
    print("✓ Extracting validation features...")
    X_val, y_val = extract_features(resnet, val_loader, device)
    
    # Train logistic regression
    print("\n✓ Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    
    print(f"\n✓ Train Accuracy: {train_acc*100:.2f}%")
    print(f"✓ Val Accuracy: {val_acc*100:.2f}%")
    
    # Save
    joblib.dump(clf, model_save_path)
    print(f"\n✓ Model saved to {model_save_path}")
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
