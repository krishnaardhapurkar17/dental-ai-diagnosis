import sys
sys.path.append('d:/college/EDAI/final project')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
from tqdm import tqdm
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

from data_loader import ClassificationDataset, SegmentationDataset
from augmentation import get_classification_val_transforms, get_segmentation_val_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}\n")

# ============================================================================
# 1. CLASSIFICATION MODELS EVALUATION
# ============================================================================

def eval_classification_models():
    print("\n" + "="*80)
    print("EVALUATING CLASSIFICATION MODELS")
    print("="*80)
    
    DATA_DIR = Path("d:/college/EDAI/final project/model1_classification/data")
    MODEL_DIR = Path("d:/college/EDAI/final project/comparative_models/model1_classification")
    
    val_transform = get_classification_val_transforms((300, 300))
    val_dataset = ClassificationDataset(DATA_DIR / "test", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(val_dataset)}")
    print(f"Classes: {val_dataset.class_names}\n")
    
    models_config = {
        "MobileNetV3-Large": {
            "path": MODEL_DIR / "mobilenetv3_best.pth",
            "arch": lambda: models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        },
        "VGG16": {
            "path": MODEL_DIR / "vgg16_best.pth",
            "arch": lambda: models.vgg16(weights='IMAGENET1K_V1')
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\nEvaluating {model_name}...")
        
        model = config["arch"]()
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
        
        state_dict = torch.load(config["path"], map_location=DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"{model_name}"):
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        results[model_name] = {
            "accuracy": acc * 100,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
        }
        
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return results

# ============================================================================
# 2. DETECTION MODELS EVALUATION
# ============================================================================

def eval_detection_models():
    print("\n" + "="*80)
    print("EVALUATING DETECTION MODELS")
    print("="*80)
    
    MODEL_DIR = Path("d:/college/EDAI/final project/comparative_models/model2_detection")
    
    models_config = {
        "YOLOv8n-Cavity": MODEL_DIR / "cavity_yolov8n" / "weights" / "best.pt",
        "YOLOv8l-Cavity": MODEL_DIR / "cavity_yolov8l" / "weights" / "best.pt",
        "YOLOv8n-OPG": MODEL_DIR / "opg_yolov8n" / "weights" / "best.pt",
        "YOLOv8l-OPG": MODEL_DIR / "opg_yolov8l" / "weights" / "best.pt"
    }
    
    results = {}
    
    for model_name, model_path in models_config.items():
        print(f"\nEvaluating {model_name}...")
        
        if not model_path.exists():
            print(f"  WARNING: Model not found at {model_path}")
            results[model_name] = {"error": "Model file not found"}
            continue
        
        try:
            model = YOLO(str(model_path))
            metrics = model.val(verbose=False)
            
            results[model_name] = {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr)
            }
            
            print(f"  mAP50:     {metrics.box.map50:.4f}")
            print(f"  mAP50-95:  {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall:    {metrics.box.mr:.4f}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results[model_name] = {"error": str(e)}
    
    return results

# ============================================================================
# 3. SEGMENTATION MODELS EVALUATION
# ============================================================================

def eval_segmentation_models():
    print("\n" + "="*80)
    print("EVALUATING SEGMENTATION MODELS")
    print("="*80)
    
    DATA_DIR = Path("d:/college/EDAI/final project/model3_segmentation/data")
    META_JSON = Path("d:/college/EDAI/final project/datasets/teeth-segmentation-on-dental-x-ray-images-DatasetNinja/meta.json")
    MODEL_DIR = Path("d:/college/EDAI/final project/comparative_models/model3_segmentation")
    
    val_transform = get_segmentation_val_transforms((512, 512))
    val_dataset = SegmentationDataset(DATA_DIR / "val" / "img", DATA_DIR / "val" / "ann", META_JSON, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(val_dataset)}\n")
    
    def compute_iou(pred, target):
        pred_tooth = (pred > 0.5).float()
        target_tooth = (target > 0).float()
        intersection = (pred_tooth * target_tooth).sum()
        union = pred_tooth.sum() + target_tooth.sum() - intersection
        return (intersection / union).item() if union > 0 else 0.0
    
    models_config = {
        "U-Net + ResNet34": {
            "path": MODEL_DIR / "unet_resnet34_best.pth",
            "arch": lambda: smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=2)
        },
        "DeepLabV3+ (ResNet50)": {
            "path": MODEL_DIR / "deeplabv3plus_best.pth",
            "arch": lambda: smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=2)
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"Evaluating {model_name}...")
        
        model = config["arch"]()
        state_dict = torch.load(config["path"], map_location=DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        
        ious = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"{model_name}"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                
                for i in range(outputs.shape[0]):
                    iou = compute_iou(outputs[i, 1], masks[i])
                    ious.append(iou)
        
        mean_iou = np.mean(ious)
        std_iou = np.std(ious)
        
        results[model_name] = {
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "min_iou": np.min(ious),
            "max_iou": np.max(ious)
        }
        
        print(f"  Mean IoU:  {mean_iou:.4f}")
        print(f"  Std IoU:   {std_iou:.4f}")
        print(f"  Min IoU:   {np.min(ious):.4f}")
        print(f"  Max IoU:   {np.max(ious):.4f}")
    
    return results

# ============================================================================
# 4. CLINICAL CLASSIFICATION MODELS EVALUATION
# ============================================================================

def eval_clinical_models():
    print("\n" + "="*80)
    print("EVALUATING CLINICAL CLASSIFICATION MODELS")
    print("="*80)
    
    DATA_DIR = Path("d:/college/EDAI/final project/datasets/normal photographs/Dental diseases_Model")
    MODEL_DIR = Path("d:/college/EDAI/final project/comparative_models/model4_clinical")
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    from torchvision.datasets import ImageFolder
    full_dataset = ImageFolder(DATA_DIR, transform=val_transforms)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {full_dataset.classes}\n")
    
    models_config = {
        "EfficientNet-B0": {
            "path": MODEL_DIR / "efficientnet_b0_best.pth",
            "arch": lambda: models.efficientnet_b0(weights='IMAGENET1K_V1')
        },
        "DenseNet121": {
            "path": MODEL_DIR / "densenet121_best.pth",
            "arch": lambda: models.densenet121(weights='IMAGENET1K_V1')
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"Evaluating {model_name}...")
        
        model = config["arch"]()
        if hasattr(model.classifier, 'in_features'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        state_dict = torch.load(config["path"], map_location=DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"{model_name}"):
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        results[model_name] = {
            "accuracy": acc * 100,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist()
        }
        
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return results

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    all_results = {}
    
    try:
        all_results["classification"] = eval_classification_models()
    except Exception as e:
        print(f"ERROR in classification evaluation: {e}")
        all_results["classification"] = {"error": str(e)}
    
    try:
        all_results["detection"] = eval_detection_models()
    except Exception as e:
        print(f"ERROR in detection evaluation: {e}")
        all_results["detection"] = {"error": str(e)}
    
    try:
        all_results["segmentation"] = eval_segmentation_models()
    except Exception as e:
        print(f"ERROR in segmentation evaluation: {e}")
        all_results["segmentation"] = {"error": str(e)}
    
    try:
        all_results["clinical"] = eval_clinical_models()
    except Exception as e:
        print(f"ERROR in clinical evaluation: {e}")
        all_results["clinical"] = {"error": str(e)}
    
    # ========================================================================
    # GENERATE SUMMARY REPORT
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*80)
    
    summary = {}
    
    # Classification
    print("\n1. CLASSIFICATION MODELS")
    print("-" * 80)
    if "error" not in all_results["classification"]:
        for model_name, metrics in all_results["classification"].items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        best_clf = max(all_results["classification"].items(), 
                      key=lambda x: x[1]['f1_score'])
        summary["classification"] = best_clf[0]
        print(f"\nBEST: {best_clf[0]} (F1: {best_clf[1]['f1_score']:.4f})")
    
    # Detection
    print("\n2. DETECTION MODELS")
    print("-" * 80)
    if "error" not in all_results["detection"]:
        for model_name, metrics in all_results["detection"].items():
            if "error" not in metrics:
                print(f"\n{model_name}:")
                print(f"  mAP50:     {metrics['mAP50']:.4f}")
                print(f"  mAP50-95:  {metrics['mAP50-95']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
        
        valid_models = {k: v for k, v in all_results["detection"].items() if "error" not in v}
        if valid_models:
            best_det = max(valid_models.items(), key=lambda x: x[1]['mAP50'])
            summary["detection"] = best_det[0]
            print(f"\nBEST: {best_det[0]} (mAP50: {best_det[1]['mAP50']:.4f})")
    
    # Segmentation
    print("\n3. SEGMENTATION MODELS")
    print("-" * 80)
    if "error" not in all_results["segmentation"]:
        for model_name, metrics in all_results["segmentation"].items():
            print(f"\n{model_name}:")
            print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
            print(f"  Std IoU:   {metrics['std_iou']:.4f}")
            print(f"  Min IoU:   {metrics['min_iou']:.4f}")
            print(f"  Max IoU:   {metrics['max_iou']:.4f}")
        
        best_seg = max(all_results["segmentation"].items(), 
                      key=lambda x: x[1]['mean_iou'])
        summary["segmentation"] = best_seg[0]
        print(f"\nBEST: {best_seg[0]} (Mean IoU: {best_seg[1]['mean_iou']:.4f})")
    
    # Clinical
    print("\n4. CLINICAL CLASSIFICATION MODELS")
    print("-" * 80)
    if "error" not in all_results["clinical"]:
        for model_name, metrics in all_results["clinical"].items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        best_clin = max(all_results["clinical"].items(), 
                       key=lambda x: x[1]['f1_score'])
        summary["clinical"] = best_clin[0]
        print(f"\nBEST: {best_clin[0]} (F1: {best_clin[1]['f1_score']:.4f})")
    
    # Final Recommendations
    print("\n" + "="*80)
    print("FINAL MODEL RECOMMENDATIONS")
    print("="*80)
    for task, model in summary.items():
        print(f"{task.upper():20} -> {model}")
    
    # Save detailed report
    report_path = Path("d:/college/EDAI/final project/evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary_path = Path("d:/college/EDAI/final project/model_recommendations.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Recommendations saved to: {summary_path}")

if __name__ == "__main__":
    main()
