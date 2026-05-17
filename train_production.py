import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def train_production_model():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_yaml = Path("d:/college/EDAI/project/datasets/Cavity Dataset/data.yaml")
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    print("\n" + "="*60)
    print("PRODUCTION CAVITY DETECTION TRAINING")
    print("="*60)
    print(f"\nData: {data_yaml}")
    print("\nStrategy for 10.76:1 imbalance (91.5% healthy):")
    print("  - cls=3.0 (heavy class loss weight for minority)")
    print("  - box=10.0 (prioritize localization accuracy)")
    print("  - Aggressive augmentation on minority class")
    print("  - Lower conf threshold for unhealthy detection")
    print("  - Extended training with patience")
    
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=640,
        batch=4,
        project=str(save_dir),
        name='cavity_production',
        patience=25,
        save=True,
        plots=True,
        device='cuda:0',
        workers=0,
        cls=3.0,
        box=10.0,
        degrees=20.0,
        translate=0.2,
        scale=0.6,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        hsv_h=0.03,
        hsv_s=0.7,
        hsv_v=0.5,
        close_mosaic=20,
        single_cls=False
    )
    
    print("\n" + "="*60)
    print("VALIDATING ON ALL SPLITS")
    print("="*60)
    
    splits = {'train': 'train', 'valid': 'val', 'test': 'test'}
    for split_name, split_key in splits.items():
        print(f"\n{split_name.upper()}:")
        metrics = model.val(data=str(data_yaml), split=split_key, workers=0, verbose=False)
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  Healthy: {metrics.box.ap50[0]:.4f}")
        print(f"  Unhealthy: {metrics.box.ap50[1]:.4f}")

if __name__ == "__main__":
    train_production_model()
