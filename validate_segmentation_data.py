import yaml
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import SegmentationDataset
from augmentation import get_segmentation_train_transforms
import numpy as np

if __name__ == '__main__':
    def load_config():
        with open("config.yaml", 'r') as f:
            return yaml.safe_load(f)

    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model3_segmentation']['data_dir']
    meta_json = base_dir / config['model3_segmentation']['meta_json']
    input_size = tuple(config['model3_segmentation']['input_size'])

    train_img_dir = data_dir / "train" / "img"
    train_ann_dir = data_dir / "train" / "ann"

    transform = get_segmentation_train_transforms(input_size)
    dataset = SegmentationDataset(train_img_dir, train_ann_dir, meta_json, transform=transform)

    print(f"Dataset size: {len(dataset)}")
    print("\nChecking first 5 samples:")

    for i in range(min(5, len(dataset))):
        img, mask = dataset[i]
        unique_classes = torch.unique(mask)
        print(f"\nSample {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Unique classes in mask: {unique_classes.tolist()}")
        print(f"  Number of teeth: {len(unique_classes) - 1 if 0 in unique_classes else len(unique_classes)}")
        
        # Check class distribution
        for cls in unique_classes:
            count = (mask == cls).sum().item()
            pct = count / mask.numel() * 100
            print(f"    Class {cls}: {count} pixels ({pct:.2f}%)")

    print("\n[OK] Data validation complete")
