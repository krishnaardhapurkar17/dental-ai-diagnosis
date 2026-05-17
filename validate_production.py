import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def validate_production():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    data_yaml = Path("d:/college/EDAI/project/datasets/Cavity Dataset/data.yaml")
    model_path = base_dir / "model2_detection/checkpoints/cavity_production/weights/best.pt"
    
    print("="*60)
    print("PRODUCTION MODEL VALIDATION")
    print("="*60)
    
    model = YOLO(str(model_path))
    
    splits = {'TRAIN': 'train', 'VALID': 'val', 'TEST': 'test'}
    for split_name, split_key in splits.items():
        print(f"\n{split_name}:")
        metrics = model.val(data=str(data_yaml), split=split_key, workers=0, verbose=False)
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  Healthy: {metrics.box.ap50[0]:.4f}")
        print(f"  Unhealthy: {metrics.box.ap50[1]:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    validate_production()
