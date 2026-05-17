import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def train_cavity_detection(config):
    """Train YOLOv8 for cavity detection (healthy/unhealthy teeth)"""
    print("\n" + "="*60)
    print("TRAINING CAVITY DETECTION MODEL (YOLOv8)")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    data_yaml = base_dir / config['model2_detection']['cavity_data_yaml']
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    epochs = config['model2_detection']['epochs']
    batch_size = config['model2_detection']['batch_size']
    img_size = config['model2_detection']['input_size']
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"\n❌ Error: Data config not found at {data_yaml}")
        print("Please ensure the Cavity Dataset is properly set up.")
        return None
    
    print(f"\n✓ Data config: {data_yaml}")
    print(f"✓ Training for {epochs} epochs")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Image size: {img_size}")
    
    # Load YOLOv8 medium model (best balance for medical imaging)
    print("\n✓ Loading YOLOv8m (medium) pretrained model...")
    model = YOLO('yolov8m.pt')
    
    # Train
    print("\n✓ Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=str(save_dir),
        name='cavity',
        patience=10,
        save=True,
        plots=True,
        device=0 if os.name != 'nt' else 'cuda:0',
        workers=0,
        cls=1.5  # Increase classification loss weight for class imbalance
    )
    
    # Get metrics
    metrics = model.val()
    
    print("\n" + "="*60)
    print("CAVITY DETECTION TRAINING COMPLETE")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def train_opg_detection(config):
    """Train YOLOv8 for OPG detection (6 dental conditions)"""
    print("\n" + "="*60)
    print("TRAINING OPG DETECTION MODEL (YOLOv8)")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    data_yaml = base_dir / config['model2_detection']['opg_data_yaml']
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    epochs = config['model2_detection']['epochs']
    batch_size = config['model2_detection']['batch_size']
    img_size = config['model2_detection']['input_size']
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"\n❌ Error: Data config not found at {data_yaml}")
        print("Please ensure the OPG Detection dataset is properly set up.")
        return None
    
    print(f"\n✓ Data config: {data_yaml}")
    print(f"✓ Training for {epochs} epochs")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Image size: {img_size}")
    
    # Load YOLOv8 medium model
    print("\n✓ Loading YOLOv8m (medium) pretrained model...")
    model = YOLO('yolov8m.pt')
    
    # Train
    print("\n✓ Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=str(save_dir),
        name='opg',
        patience=10,
        save=True,
        plots=True,
        device=0 if os.name != 'nt' else 'cuda:0',
        workers=0  # Disable multiprocessing
    )
    
    # Get metrics
    metrics = model.val()
    
    print("\n" + "="*60)
    print("OPG DETECTION TRAINING COMPLETE")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def main():
    print("\n" + "="*60)
    print("DENTAL DETECTION TRAINING (YOLOv8)")
    print("="*60)
    
    # Load config
    try:
        config = load_config()
        print("✓ Configuration loaded from config.yaml")
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return
    
    # Train Cavity Detection
    print("\n" + "="*60)
    print("TASK A: CAVITY DETECTION")
    print("="*60)
    try:
        cavity_metrics = train_cavity_detection(config)
    except Exception as e:
        print(f"\n❌ Error training cavity detection: {e}")
        cavity_metrics = None
    
    # Train OPG Detection
    print("\n" + "="*60)
    print("TASK B: OPG DETECTION")
    print("="*60)
    try:
        opg_metrics = train_opg_detection(config)
    except Exception as e:
        print(f"\n❌ Error training OPG detection: {e}")
        opg_metrics = None
    
    # Summary
    print("\n" + "="*60)
    print("✅ DETECTION TRAINING COMPLETE!")
    print("="*60)
    
    if cavity_metrics:
        print(f"\nCavity Detection:")
        print(f"  mAP50: {cavity_metrics.box.map50:.4f}")
        print(f"  mAP50-95: {cavity_metrics.box.map:.4f}")
    
    if opg_metrics:
        print(f"\nOPG Detection:")
        print(f"  mAP50: {opg_metrics.box.map50:.4f}")
        print(f"  mAP50-95: {opg_metrics.box.map:.4f}")
    
    print("\nNext step: Run python eval_detection.py")

if __name__ == "__main__":
    main()
