import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def retrain_cavity():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_yaml = base_dir / config['model2_detection']['cavity_data_yaml']
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    print("\n" + "="*60)
    print("RETRAINING CAVITY DETECTION (Class-weighted + Augmentation)")
    print("="*60)
    print(f"\nData: {data_yaml}")
    print("Modifications:")
    print("  - cls=2.0 (stronger class loss weight)")
    print("  - Enhanced augmentation for minority class")
    print("  - Focal loss via cls weight")
    
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=4,
        project=str(save_dir),
        name='cavity_balanced',
        patience=15,
        save=True,
        plots=True,
        device='cuda:0',
        workers=0,
        cls=2.0,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    metrics = model.val()
    
    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    if hasattr(metrics.box, 'ap50'):
        print("\nPer-class AP50:")
        for i, ap in enumerate(metrics.box.ap50):
            print(f"  {model.names[i]}: {ap:.4f}")

if __name__ == "__main__":
    retrain_cavity()
