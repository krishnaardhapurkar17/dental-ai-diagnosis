import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def final_evaluation():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    print("="*60)
    print("FINAL DENTAL AI EVALUATION")
    print("="*60)
    
    # Cavity Detection (best model: cavity_production)
    print("\nCavity Detection (cavity_production):")
    cavity_path = save_dir / "cavity_production" / "weights" / "best.pt"
    cavity_data = base_dir / config['model2_detection']['cavity_data_yaml']
    
    cavity_model = YOLO(str(cavity_path))
    cavity_metrics = cavity_model.val(data=str(cavity_data), split='test', workers=0, verbose=False)
    
    print(f"  mAP50: {cavity_metrics.box.map50:.4f}")
    print(f"  mAP50-95: {cavity_metrics.box.map:.4f}")
    print(f"  Healthy teeth: {cavity_metrics.box.ap50[0]:.4f}")
    print(f"  Unhealthy teeth: {cavity_metrics.box.ap50[1]:.4f}")
    
    # OPG Detection
    print("\nOPG Detection (opg3):")
    opg_path = save_dir / "opg3" / "weights" / "best.pt"
    opg_data = base_dir / config['model2_detection']['opg_data_yaml']
    
    opg_model = YOLO(str(opg_path))
    opg_metrics = opg_model.val(data=str(opg_data), split='test', workers=0, verbose=False)
    
    print(f"  mAP50: {opg_metrics.box.map50:.4f}")
    print(f"  mAP50-95: {opg_metrics.box.map:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    cavity_pass = cavity_metrics.box.map50 > 0.75
    opg_pass = opg_metrics.box.map50 > 0.75
    
    print(f"Cavity Detection: {cavity_metrics.box.map50:.4f} {'✓' if cavity_pass else '❌'} (target: >0.75)")
    print(f"OPG Detection: {opg_metrics.box.map50:.4f} {'✓' if opg_pass else '❌'} (target: >0.75)")
    
    if opg_pass and not cavity_pass:
        print("\nStatus: Partial success - OPG detection meets target")
        print("Issue: Cavity detection limited by severe test set class imbalance")
        print("Recommendation: Deploy OPG model, retrain cavity with balanced dataset")
    elif opg_pass and cavity_pass:
        print("\nStatus: Full success - Both models meet targets")
    else:
        print("\nStatus: Needs improvement")

if __name__ == "__main__":
    final_evaluation()