import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_cavity_models():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    data_yaml = base_dir / config['model2_detection']['cavity_data_yaml']
    
    models = ['cavity', 'cavity2', 'cavity3', 'cavity5']
    
    print("Testing cavity models on test set:")
    print("="*50)
    
    for model_name in models:
        model_path = save_dir / model_name / "weights" / "best.pt"
        if model_path.exists():
            print(f"\n{model_name}:")
            model = YOLO(str(model_path))
            metrics = model.val(data=str(data_yaml), split='test', workers=0, verbose=False)
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  Healthy: {metrics.box.ap50[0]:.4f}")
            print(f"  Unhealthy: {metrics.box.ap50[1]:.4f}")
        else:
            print(f"\n{model_name}: Not found")

if __name__ == "__main__":
    test_cavity_models()