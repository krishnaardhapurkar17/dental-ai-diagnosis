import os
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from data_loader import SegmentationDataset
from augmentation import get_segmentation_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def compute_per_class_iou(pred, target, num_classes):
    """Compute IoU for each class"""
    ious = {}
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(1, num_classes + 1):  # Skip background (0)
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            ious[cls] = None  # Class not present
        else:
            ious[cls] = (intersection / union).item()
    
    return ious

def visualize_predictions(model, dataset, device, num_samples=5, save_dir=None):
    """Visualize sample predictions with ground truth overlay"""
    model.eval()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        image, mask = dataset[idx]
        
        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_batch)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(image_np)
        axes[1].imshow(mask_np, alpha=0.5, cmap='tab20')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(image_np)
        axes[2].imshow(pred, alpha=0.5, cmap='tab20')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    print("\n" + "="*60)
    print("EVALUATING SEGMENTATION MODEL")
    print("="*60)
    
    # Load config
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model3_segmentation']['data_dir']
    model_save_path = base_dir / config['model3_segmentation']['model_save_path']
    meta_json = base_dir / config['model3_segmentation']['meta_json']
    outputs_dir = base_dir / config['evaluation']['outputs_dir'] / "segmentation"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = config['model3_segmentation']['num_classes']
    input_size = tuple(config['model3_segmentation']['input_size'])
    batch_size = config['model3_segmentation']['batch_size']
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Check if model exists
    if not model_save_path.exists():
        print(f"\n❌ Error: Model not found at {model_save_path}")
        print("Please run: python train_segmentation.py")
        return
    
    # Load test dataset
    test_img_dir = data_dir / "test" / "img"
    test_ann_dir = data_dir / "test" / "ann"
    
    if not test_img_dir.exists():
        print(f"\n❌ Error: Test data not found at {test_img_dir}")
        return
    
    val_transform = get_segmentation_val_transforms(input_size)
    test_dataset = SegmentationDataset(test_img_dir, test_ann_dir, meta_json, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n✓ Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"\n✓ Loading model from {model_save_path}")
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights=None,
        in_channels=3,
        classes=num_classes + 1
    )
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("\n✓ Running evaluation with Test Time Augmentation...")
    all_class_ious = {cls: [] for cls in range(1, num_classes + 1)}
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            # Original prediction
            outputs = model(images)
            
            # Horizontal flip prediction
            outputs_hflip = model(torch.flip(images, dims=[3]))
            outputs_hflip = torch.flip(outputs_hflip, dims=[3])
            
            # Average predictions (TTA)
            outputs = (outputs + outputs_hflip) / 2
            preds = outputs.argmax(dim=1)
            
            # Compute per-class IoU for each sample
            for pred, mask in zip(preds, masks):
                class_ious = compute_per_class_iou(pred, mask, num_classes)
                for cls, iou in class_ious.items():
                    if iou is not None:
                        all_class_ious[cls].append(iou)
    
    # Compute mean IoU per class
    print("\n" + "="*60)
    print("PER-TOOTH IoU (FDI Numbering)")
    print("="*60)
    
    per_class_mean_iou = {}
    for cls in range(1, num_classes + 1):
        if all_class_ious[cls]:
            mean_iou = np.mean(all_class_ious[cls])
            per_class_mean_iou[cls] = mean_iou
            print(f"Tooth {cls}: {mean_iou:.4f} (n={len(all_class_ious[cls])})")
        else:
            per_class_mean_iou[cls] = None
            print(f"Tooth {cls}: Not present in test set")
    
    # Overall mean IoU
    valid_ious = [iou for iou in per_class_mean_iou.values() if iou is not None]
    overall_mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    print("\n" + "="*60)
    print(f"OVERALL MEAN IoU: {overall_mean_iou:.4f}")
    print("="*60)
    
    if overall_mean_iou > 0.70:
        print("✓ Target achieved (mean IoU > 0.70)")
    else:
        print("⚠️  Below target (mean IoU should be > 0.70)")
    
    # Save metrics
    metrics_data = []
    for cls in range(1, num_classes + 1):
        metrics_data.append({
            'Tooth': cls,
            'IoU': per_class_mean_iou[cls] if per_class_mean_iou[cls] is not None else 'N/A',
            'Samples': len(all_class_ious[cls])
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = outputs_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Visualize sample predictions
    print("\n✓ Generating sample visualizations...")
    samples_dir = outputs_dir / "samples"
    visualize_predictions(model, test_dataset, device, num_samples=5, save_dir=samples_dir)
    print(f"✓ Sample predictions saved to {samples_dir}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"Overall mean IoU: {overall_mean_iou:.4f}")
    print(f"Results saved to: {outputs_dir}")

if __name__ == "__main__":
    main()
