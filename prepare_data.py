import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def prepare_classification_data(config):
    """Prepare classification dataset (OPG Classification)"""
    print("\n" + "="*60)
    print("PREPARING CLASSIFICATION DATASET")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    source_dir = base_dir / config['model1_classification']['data_source']
    target_dir = base_dir / config['model1_classification']['data_dir']
    classes = config['model1_classification']['classes']
    random_seed = config['project']['random_seed']
    
    # Collect all image paths with labels
    images = []
    labels = []
    
    for class_name in classes:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] {class_dir} not found, skipping")
            continue
        
        for img_path in class_dir.glob("*.jpg"):
            try:
                # Validate image can be opened
                from PIL import Image
                Image.open(img_path).verify()
                images.append(img_path)
                labels.append(class_name)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image: {img_path}")
    
    print(f"\n[OK] Total valid images found: {len(images)}")
    print(f"[OK] Class distribution: {dict(Counter(labels))}")
    
    # Check for duplicate filenames
    filenames = [img.name for img in images]
    duplicates = [name for name in set(filenames) if filenames.count(name) > 1]
    if duplicates:
        print(f"[WARNING] Found {len(duplicates)} duplicate filenames")
    
    # Stratified split: 70% train, 20% val, 10% test
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=random_seed
    )
    
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.33, stratify=temp_labels, random_state=random_seed
    )
    
    print(f"\n[OK] Split complete:")
    print(f"   Train: {len(train_imgs)} images")
    print(f"   Val:   {len(val_imgs)} images")
    print(f"   Test:  {len(test_imgs)} images")
    
    # Copy files to target directories
    for split_name, split_imgs, split_labels in [
        ("train", train_imgs, train_labels),
        ("val", val_imgs, val_labels),
        ("test", test_imgs, test_labels)
    ]:
        for img_path, label in zip(split_imgs, split_labels):
            target_class_dir = target_dir / split_name / label
            target_class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target_class_dir / img_path.name)
        
        print(f"[OK] {split_name.capitalize()} set copied: {dict(Counter(split_labels))}")
    
    # Save split statistics
    stats_file = target_dir / "split_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("CLASSIFICATION DATASET SPLIT STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Train: {len(train_imgs)} ({len(train_imgs)/len(images)*100:.1f}%)\n")
        f.write(f"Val: {len(val_imgs)} ({len(val_imgs)/len(images)*100:.1f}%)\n")
        f.write(f"Test: {len(test_imgs)} ({len(test_imgs)/len(images)*100:.1f}%)\n\n")
        f.write(f"Train distribution: {dict(Counter(train_labels))}\n")
        f.write(f"Val distribution: {dict(Counter(val_labels))}\n")
        f.write(f"Test distribution: {dict(Counter(test_labels))}\n")
    
    print(f"\n[OK] Classification statistics saved to {stats_file}")

def prepare_segmentation_data(config):
    """Prepare segmentation dataset (Teeth Segmentation)"""
    print("\n" + "="*60)
    print("PREPARING SEGMENTATION DATASET")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    source_dir = base_dir / config['model3_segmentation']['data_source']
    target_dir = base_dir / config['model3_segmentation']['data_dir']
    random_seed = config['project']['random_seed']
    
    img_dir = source_dir / "img"
    ann_dir = source_dir / "ann"
    
    # Get all image files that have corresponding annotations
    image_files = []
    for img_file in sorted(img_dir.glob("*.jpg")):
        ann_file = ann_dir / f"{img_file.name}.json"
        if ann_file.exists():
            try:
                # Validate image
                from PIL import Image
                Image.open(img_file).verify()
                image_files.append(img_file.name)
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image: {img_file.name}")
        else:
            print(f"[WARNING] Missing annotation for {img_file.name}")
    
    print(f"\n[OK] Total valid image-annotation pairs: {len(image_files)}")
    
    # Random split: 70% train, 20% val, 10% test
    train_files, temp_files = train_test_split(
        image_files, test_size=0.3, random_state=random_seed
    )
    
    val_files, test_files = train_test_split(
        temp_files, test_size=0.33, random_state=random_seed
    )
    
    print(f"\n[OK] Split complete:")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val:   {len(val_files)} images")
    print(f"   Test:  {len(test_files)} images")
    
    # Copy files to target directories
    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]:
        target_img_dir = target_dir / split_name / "img"
        target_ann_dir = target_dir / split_name / "ann"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_ann_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in split_files:
            # Copy image
            shutil.copy2(img_dir / filename, target_img_dir / filename)
            # Copy annotation
            shutil.copy2(ann_dir / f"{filename}.json", target_ann_dir / f"{filename}.json")
        
        print(f"[OK] {split_name.capitalize()} set copied: {len(split_files)} pairs")
    
    # Save split statistics
    stats_file = target_dir / "split_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("SEGMENTATION DATASET SPLIT STATISTICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total image-annotation pairs: {len(image_files)}\n")
        f.write(f"Train: {len(train_files)} ({len(train_files)/len(image_files)*100:.1f}%)\n")
        f.write(f"Val: {len(val_files)} ({len(val_files)/len(image_files)*100:.1f}%)\n")
        f.write(f"Test: {len(test_files)} ({len(test_files)/len(image_files)*100:.1f}%)\n")
    
    print(f"\n[OK] Segmentation statistics saved to {stats_file}")

def main():
    """Main function to prepare all datasets"""
    print("\n" + "="*60)
    print("DENTAL AI DATASET PREPARATION")
    print("="*60)
    
    try:
        config = load_config()
        print("[OK] Configuration loaded from config.yaml")
    except Exception as e:
        print(f"[ERROR] Error loading config.yaml: {e}")
        return
    
    # Prepare classification dataset
    try:
        prepare_classification_data(config)
    except Exception as e:
        print(f"\n[ERROR] Error preparing classification dataset: {e}")
    
    # Prepare segmentation dataset
    try:
        prepare_segmentation_data(config)
    except Exception as e:
        print(f"\n[ERROR] Error preparing segmentation dataset: {e}")
    
    print("\n" + "="*60)
    print("[SUCCESS] DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python train_classification.py")
    print("2. Run: python train_detection.py")
    print("3. Run: python train_segmentation.py")

if __name__ == "__main__":
    main()
