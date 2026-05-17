import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import segmentation_models_pytorch as smp
from ultralytics import YOLO

from data_loader import ClassificationDataset, SegmentationDataset
from augmentation import get_classification_val_transforms, get_segmentation_val_transforms

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def compute_iou(pred, target):
    pred_tooth = (pred > 0).float()
    target_tooth = (target > 0).float()
    intersection = (pred_tooth * target_tooth).sum()
    union = pred_tooth.sum() + target_tooth.sum() - intersection
    return (intersection / union).item() if union > 0 else 0.0

def eval_model1_classification(config, device, outputs_dir):
    print("\n" + "="*60)
    print("MODEL 1: OPG CLASSIFICATION (ResNet18)")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model1_classification']['data_dir']
    model_path = base_dir / config['model1_classification']['model_save_path']
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    test_dir = data_dir / "test"
    if not test_dir.exists():
        print(f"❌ Test data not found: {test_dir}")
        return None
    
    val_transform = get_classification_val_transforms(tuple(config['model1_classification']['input_size']))
    test_dataset = ClassificationDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['model1_classification']['batch_size'], shuffle=False, num_workers=0)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=config['model1_classification']['dropout']),
        nn.Linear(model.fc.in_features, config['model1_classification']['num_classes'])
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    report = classification_report(all_labels, all_preds, target_names=test_dataset.class_names, digits=3, output_dict=True)
    
    metrics = {
        'model': 'Model1_OPG_Classification',
        'architecture': 'ResNet18',
        'test_samples': len(test_dataset),
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    print(f"✓ Accuracy: {accuracy:.2f}%")
    print(f"✓ F1-Score: {metrics['f1_score']:.3f}")
    
    return metrics

def eval_model2_detection(config, outputs_dir):
    print("\n" + "="*60)
    print("MODEL 2: DETECTION (YOLOv8)")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    save_dir = base_dir / config['model2_detection']['model_save_dir']
    
    results = []
    
    # Cavity Detection
    cavity_model = save_dir / "cavity" / "weights" / "best.pt"
    if cavity_model.exists():
        print("\n✓ Evaluating Cavity Detection...")
        model = YOLO(str(cavity_model))
        cavity_dir = base_dir / "datasets" / "normal photographs" / "Cavity Dataset"
        cavity_yaml = cavity_dir / "data_eval.yaml"
        if not cavity_yaml.exists():
            cavity_yaml = cavity_dir / "data.yaml"
        metrics = model.val(data=str(cavity_yaml), split='test', verbose=False, imgsz=640)
        results.append({
            'model': 'Model2_Cavity_Detection',
            'architecture': 'YOLOv8m',
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        })
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"❌ Cavity model not found: {cavity_model}")
    
    # OPG Detection
    opg_model_paths = [
        save_dir / "opg3" / "weights" / "best.pt",
        save_dir / "opg2" / "weights" / "best.pt",
        save_dir / "opg" / "weights" / "best.pt"
    ]
    opg_model = None
    for path in opg_model_paths:
        if path.exists():
            opg_model = path
            break
    
    if opg_model:
        print("\n✓ Evaluating OPG Detection...")
        model = YOLO(str(opg_model))
        opg_dir = base_dir / "datasets" / "xrays" / "Dental OPG XRAY Dataset" / "Dental OPG (Object Detection)" / "Augmented Dataset"
        metrics = model.val(data=str(opg_dir / "data.yaml"), split='test', verbose=False, imgsz=640)
        results.append({
            'model': 'Model2_OPG_Detection',
            'architecture': 'YOLOv8m',
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        })
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print(f"❌ OPG detection model not found in any location")
    
    return results

def eval_model3_segmentation(config, device, outputs_dir):
    print("\n" + "="*60)
    print("MODEL 3: SEGMENTATION (U-Net + EfficientNet-B3)")
    print("="*60)
    
    base_dir = Path(config['project']['base_dir'])
    data_dir = base_dir / config['model3_segmentation']['data_dir']
    model_path = base_dir / config['model3_segmentation']['model_save_path']
    meta_json = base_dir / config['model3_segmentation']['meta_json']
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    test_img_dir = data_dir / "test" / "img"
    test_ann_dir = data_dir / "test" / "ann"
    if not test_img_dir.exists():
        print(f"❌ Test data not found: {test_img_dir}")
        return None
    
    val_transform = get_segmentation_val_transforms(tuple(config['model3_segmentation']['input_size']))
    test_dataset = SegmentationDataset(test_img_dir, test_ann_dir, meta_json, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['model3_segmentation']['batch_size'], shuffle=False, num_workers=0)
    
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights=None,
        in_channels=3,
        classes=config['model3_segmentation']['num_classes'] + 1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    ious = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for i in range(preds.shape[0]):
                iou = compute_iou(preds[i], masks[i])
                ious.append(iou)
    
    mean_iou = np.mean(ious)
    
    metrics = {
        'model': 'Model3_Segmentation',
        'architecture': 'UNet_EfficientNetB3',
        'test_samples': len(test_dataset),
        'mean_iou': mean_iou,
        'median_iou': np.median(ious),
        'std_iou': np.std(ious)
    }
    
    print(f"✓ Mean IoU: {mean_iou:.4f}")
    print(f"✓ Median IoU: {metrics['median_iou']:.4f}")
    
    return metrics

def eval_model4_clinical(outputs_dir):
    print("\n" + "="*60)
    print("MODEL 4: CLINICAL PHOTO CLASSIFICATION (ResNet50)")
    print("="*60)
    
    data_dir = r'D:\college\EDAI\final project\datasets\normal photographs\Dental diseases_Model'
    model_path = 'model4_clinical_classification/best_clinical_photo_model.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not os.path.exists(data_dir):
        print(f"❌ Data not found: {data_dir}")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    num_classes = len(full_dataset.classes)
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    report = classification_report(all_labels, all_preds, target_names=full_dataset.classes, digits=3, output_dict=True)
    
    metrics = {
        'model': 'Model4_Clinical_Photo_Classification',
        'architecture': 'ResNet50',
        'test_samples': len(test_dataset),
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    print(f"✓ Accuracy: {accuracy:.2f}%")
    print(f"✓ F1-Score: {metrics['f1_score']:.3f}")
    
    return metrics

def main():
    print("\n" + "="*60)
    print("UNIFIED EVALUATION: ALL 4 MODELS")
    print("="*60)
    
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    outputs_dir = base_dir / config['evaluation']['outputs_dir'] / "unified_eval"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    all_metrics = []
    
    # Model 1: Classification
    try:
        m1 = eval_model1_classification(config, device, outputs_dir)
        if m1:
            all_metrics.append(m1)
    except Exception as e:
        print(f"❌ Model 1 error: {e}")
    
    # Model 2: Detection
    try:
        m2_list = eval_model2_detection(config, outputs_dir)
        if m2_list:
            all_metrics.extend(m2_list)
    except Exception as e:
        print(f"❌ Model 2 error: {e}")
    
    # Model 3: Segmentation
    try:
        m3 = eval_model3_segmentation(config, device, outputs_dir)
        if m3:
            all_metrics.append(m3)
    except Exception as e:
        print(f"❌ Model 3 error: {e}")
    
    # Model 4: Clinical Photos
    try:
        m4 = eval_model4_clinical(outputs_dir)
        if m4:
            all_metrics.append(m4)
    except Exception as e:
        print(f"❌ Model 4 error: {e}")
    
    # Save results
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = outputs_dir / "all_models_metrics.csv"
        try:
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Results saved: {csv_path}")
        except PermissionError:
            csv_path = outputs_dir / f"all_models_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Results saved: {csv_path}")
        
        print("\n" + "="*60)
        print("SUMMARY: ALL MODELS")
        print("="*60)
        print(df.to_string(index=False))
    else:
        print("\n❌ No metrics collected")

if __name__ == "__main__":
    main()
