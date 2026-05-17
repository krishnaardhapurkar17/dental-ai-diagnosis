import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import ClassificationDataset
from augmentation import get_classification_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def main():
    print("\n" + "="*60)
    print("EVALUATING CLASSIFICATION MODEL")
    print("="*60)
    
    # Load config
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model1_classification']['data_dir']
    model_save_path = base_dir / config['model1_classification']['model_save_path']
    outputs_dir = base_dir / config['evaluation']['outputs_dir'] / "classification"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = config['model1_classification']['num_classes']
    input_size = tuple(config['model1_classification']['input_size'])
    batch_size = config['model1_classification']['batch_size']
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Check if model exists
    if not model_save_path.exists():
        print(f"\n❌ Error: Model not found at {model_save_path}")
        print("Please run: python train_classification.py")
        return
    
    # Load test dataset
    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"\n❌ Error: Test data not found at {test_dir}")
        return
    
    val_transform = get_classification_val_transforms(input_size)
    test_dataset = ClassificationDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {test_dataset.class_names}")
    
    # Load model
    print(f"\n✓ Loading model from {model_save_path}")
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=config['model1_classification']['dropout']),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("\n✓ Running evaluation...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n✓ Overall Test Accuracy: {accuracy:.2f}%")
    
    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=test_dataset.class_names,
        digits=3,
        output_dict=True
    )
    
    # Print and save metrics
    metrics_data = []
    for class_name in test_dataset.class_names:
        metrics = report[class_name]
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = metrics['support']
        
        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        print(f"  Support:   {int(support)}")
        
        if f1 < 0.7:
            print(f"  ⚠️  WARNING: F1-score below 0.7 - needs attention!")
        
        metrics_data.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': int(support)
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = outputs_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Confusion matrix
    print("\n✓ Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=test_dataset.class_names,
        yticklabels=test_dataset.class_names
    )
    plt.title(f'Confusion Matrix - Test Accuracy: {accuracy:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = outputs_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {outputs_dir}")
    
    # Flag classes needing attention
    low_f1_classes = [m['Class'] for m in metrics_data if m['F1-Score'] < 0.7]
    if low_f1_classes:
        print(f"\n⚠️  Classes needing attention (F1 < 0.7): {', '.join(low_f1_classes)}")
    else:
        print("\n✓ All classes have F1-score >= 0.7")

if __name__ == "__main__":
    main()
