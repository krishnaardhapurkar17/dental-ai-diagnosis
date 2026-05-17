import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import json

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def eval_cavity_detection(config):
    """Evaluate cavity detection model on test set"""
    print("\n" + "="*60)
    print("EVALUATING CAVITY DETECTION MODEL")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    model_path = save_dir / "cavity5" / "weights" / "best.pt"
    data_yaml = base_dir / config['model2_detection']['cavity_data_yaml']
    
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please run: python train_detection.py")
        return None
    
    print(f"\n✓ Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Validate on test set
    print("\n✓ Running validation on test set...")
    metrics = model.val(data=str(data_yaml), split='test', workers=0)
    
    print("\n" + "="*60)
    print("CAVITY DETECTION TEST RESULTS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Per-class AP
    if hasattr(metrics.box, 'ap_class_index'):
        print("\nPer-class Average Precision (AP50):")
        for i, ap in enumerate(metrics.box.ap50):
            class_name = model.names[i] if i < len(model.names) else f"Class {i}"
            print(f"  {class_name}: {ap:.4f}")
    
    return metrics

def eval_opg_detection(config):
    """Evaluate OPG detection model on test set"""
    print("\n" + "="*60)
    print("EVALUATING OPG DETECTION MODEL")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    model_path = save_dir / "opg3" / "weights" / "best.pt"
    data_yaml = base_dir / config['model2_detection']['opg_data_yaml']
    
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please run: python train_detection.py")
        return None
    
    print(f"\n✓ Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Validate on test set
    print("\n✓ Running validation on test set...")
    metrics = model.val(data=str(data_yaml), split='test', workers=0)
    
    print("\n" + "="*60)
    print("OPG DETECTION TEST RESULTS")
    print("="*60)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Per-class AP
    if hasattr(metrics.box, 'ap_class_index'):
        print("\nPer-class Average Precision (AP50):")
        for i, ap in enumerate(metrics.box.ap50):
            class_name = model.names[i] if i < len(model.names) else f"Class {i}"
            print(f"  {class_name}: {ap:.4f}")
    
    return metrics

def eval_bitewing_caries(config):
    """Evaluate cavity model on independent Bitewing Caries dataset (clinical validation)"""
    print("\n" + "="*60)
    print("CLINICAL VALIDATION: BITEWING CARIES DATASET")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    model_path = save_dir / "cavity5" / "weights" / "best.pt"
    
    bitewing_images = base_dir / config['evaluation']['bitewing_images']
    bitewing_annotations = base_dir / config['evaluation']['bitewing_annotations']
    
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        return None
    
    if not bitewing_images.exists():
        print(f"\n❌ Error: Bitewing images not found at {bitewing_images}")
        return None
    
    if not bitewing_annotations.exists():
        print(f"\n❌ Error: Bitewing annotations not found at {bitewing_annotations}")
        return None
    
    print(f"\n✓ Loading cavity detection model...")
    model = YOLO(str(model_path))
    
    # Load COCO annotations
    print(f"✓ Loading COCO annotations from {bitewing_annotations}")
    with open(bitewing_annotations, 'r') as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    
    print(f"✓ Found {num_images} images with {num_annotations} annotations")
    
    # Run inference on bitewing images
    print("\n✓ Running inference on bitewing dataset...")
    results = model.predict(
        source=str(bitewing_images),
        save=True,
        project=str(base_dir / config['evaluation']['outputs_dir']),
        name='bitewing_predictions',
        conf=0.25
    )
    
    print("\n" + "="*60)
    print("BITEWING CARIES EVALUATION COMPLETE")
    print("="*60)
    print(f"Processed {len(results)} images")
    print(f"Predictions saved to: {base_dir / config['evaluation']['outputs_dir'] / 'bitewing_predictions'}")
    print("\nNote: For detailed metrics, manual review of predictions is recommended")
    print("as this is an independent clinical validation dataset.")
    
    return results

def main():
    print("\n" + "="*60)
    print("DENTAL DETECTION EVALUATION")
    print("="*60)
    
    # Load config
    try:
        config = load_config()
        print("✓ Configuration loaded from config.yaml")
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return
    
    # Evaluate Cavity Detection
    print("\n" + "="*60)
    print("TASK A: CAVITY DETECTION")
    print("="*60)
    try:
        cavity_metrics = eval_cavity_detection(config)
    except Exception as e:
        print(f"\n❌ Error evaluating cavity detection: {e}")
        cavity_metrics = None
    
    # Evaluate OPG Detection
    print("\n" + "="*60)
    print("TASK B: OPG DETECTION")
    print("="*60)
    try:
        opg_metrics = eval_opg_detection(config)
    except Exception as e:
        print(f"\n❌ Error evaluating OPG detection: {e}")
        opg_metrics = None
    
    # Clinical Validation on Bitewing Dataset
    print("\n" + "="*60)
    print("CLINICAL VALIDATION")
    print("="*60)
    try:
        bitewing_results = eval_bitewing_caries(config)
    except Exception as e:
        print(f"\n❌ Error on bitewing evaluation: {e}")
        bitewing_results = None
    
    # Summary
    print("\n" + "="*60)
    print("✅ DETECTION EVALUATION COMPLETE!")
    print("="*60)
    
    if cavity_metrics:
        print(f"\nCavity Detection (Test Set):")
        print(f"  mAP50: {cavity_metrics.box.map50:.4f}")
        print(f"  mAP50-95: {cavity_metrics.box.map:.4f}")
        if cavity_metrics.box.map50 > 0.75:
            print("  ✓ Target achieved (mAP50 > 0.75)")
        else:
            print("  ⚠️  Below target (mAP50 should be > 0.75)")
    
    if opg_metrics:
        print(f"\nOPG Detection (Test Set):")
        print(f"  mAP50: {opg_metrics.box.map50:.4f}")
        print(f"  mAP50-95: {opg_metrics.box.map:.4f}")
        if opg_metrics.box.map50 > 0.75:
            print("  ✓ Target achieved (mAP50 > 0.75)")
        else:
            print("  ⚠️  Below target (mAP50 should be > 0.75)")
    
    if bitewing_results:
        print(f"\nBitewing Caries (Clinical Validation):")
        print(f"  Processed {len(bitewing_results)} images")
        print("  Review predictions in outputs/bitewing_predictions/")

if __name__ == "__main__":
    main()
