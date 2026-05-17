# Code Review Graph - Dental AI Project

## Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DENTAL AI PROJECT                                │
│                    Multi-Task Dental Diagnosis System                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼────────┐            ┌────────▼────────┐
            │   DATA LAYER   │            │  ANALYSIS LAYER │
            └───────┬────────┘            └────────┬────────┘
                    │                               │
        ┌───────────┴───────────┐          ┌───────▼────────┐
        │                       │          │ eda_dental_    │
        │   datasets/           │          │ dataset.py     │
        │   (7 datasets)        │          └────────────────┘
        │                       │
        └───────┬───────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼───┐  ┌───▼───┐  ┌───▼───┐
│Model 1│  │Model 2│  │Model 3│
│Class. │  │Detect.│  │Segm.  │
└───────┘  └───────┘  └───────┘
```

## 1. Data Layer Architecture

### Dataset Hierarchy
```
datasets/
├── normal photographs/
│   ├── Calculus/                    [1,296 images] → NO ANNOTATIONS
│   │   └── Task: Classification (Calculus Detection)
│   │
│   └── Cavity Dataset/              [418 images] → YOLO Format
│       ├── train/ (287)             ✓ SPLIT READY
│       ├── valid/ (93)
│       ├── test/ (38)
│       └── Task: Object Detection (healthy/unhealthy teeth)
│
├── teeth-segmentation-on-dental-x-ray-images-DatasetNinja/
│   ├── ds/
│   │   ├── img/ (598)               [598 images] → Supervisely JSON
│   │   └── ann/ (598)               ✓ RICHEST ANNOTATIONS (32 tooth classes)
│   └── Task: Semantic Segmentation (Per-tooth masks)
│
└── xrays/
    ├── Dental caries in bitewing radiographs/
    │   ├── images/ (100)            [100 images] → COCO JSON
    │   └── Task: Object Detection   ⚠ TEST-ONLY (no training)
    │
    └── Dental OPG XRAY Dataset/
        ├── Dental OPG (Classification)/
        │   ├── BDC-BDR/ (52)        [517 images] → Folder Labels
        │   ├── Caries/ (119)        ⚠ CLASS IMBALANCE (13:223 ratio)
        │   ├── Fractured Teeth/ (13)
        │   ├── Healthy Teeth/ (223)
        │   ├── Impacted teeth/ (87)
        │   └── Infection/ (23)
        │
        └── Dental OPG (Object Detection)/
            ├── Original Dataset/    [231 images] → YOLO Format
            └── Augmented Dataset/   [604 images] → YOLO Format
                ├── train/ (558)     ✓ SPLIT READY (2.6x augmentation)
                ├── val/ (23)
                └── test/ (23)
```

### Data Flow Diagram
```
┌──────────────────────────────────────────────────────────────────────┐
│                         RAW DATASETS                                  │
│  7 datasets | 2 modalities | 3 task types | ~3,764 total images     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        ┌───────▼────────┐       ┌───────▼────────┐
        │  Pre-split     │       │  Needs Split   │
        │  (2 datasets)  │       │  (5 datasets)  │
        └───────┬────────┘       └───────┬────────┘
                │                        │
                │                ┌───────▼────────┐
                │                │ prepare_data.py│
                │                │ (stratified    │
                │                │  70/20/10)     │
                │                └───────┬────────┘
                │                        │
        ┌───────┴────────────────────────┴────────┐
        │                                          │
┌───────▼────────┐  ┌──────────────┐  ┌──────────▼─────┐
│ model1_        │  │ model2_      │  │ model3_        │
│ classification/│  │ detection/   │  │ segmentation/  │
│ data/          │  │ data/        │  │ data/          │
└────────────────┘  └──────────────┘  └────────────────┘
```

## 2. Code Module Analysis

### Module: `eda_dental_dataset.py`
**Purpose**: Comprehensive exploratory data analysis and validation

**Class Structure**:
```
DentalDatasetEDA
├── __init__(base_path)
├── count_files(directory, extensions)
├── get_image_dimensions(directory, extensions, sample_size=50)
├── analyze_yolo_labels(directory, sample_size=100)
├── analyze_calculus()
├── analyze_cavity_dataset()
├── analyze_teeth_segmentation()
├── analyze_bitewing_caries()
├── analyze_opg_classification()
├── analyze_opg_detection_original()
├── analyze_opg_detection_augmented()
├── generate_summary_report()
├── create_summary_dataframe()
├── create_visualizations()
└── generate_key_insights()
```

**Dependencies**:
```
External:
├── os, json, pathlib
├── cv2, PIL.Image
├── numpy, pandas
├── matplotlib.pyplot, seaborn
└── collections (defaultdict, Counter)

Internal:
└── None (standalone analysis module)
```

**Data Flow**:
```
datasets/ → DentalDatasetEDA → Analysis Results
                              ├── Console Report
                              ├── dataset_summary.csv
                              ├── dataset_overview.png
                              ├── opg_class_imbalance.png
                              ├── opg_augmentation_split.png
                              └── cavity_dataset_split.png
```

**Critical Functions**:
1. `analyze_yolo_labels()`: Validates YOLO annotation format, counts bboxes, class distribution
2. `get_image_dimensions()`: Samples images to detect resolution inconsistencies
3. `generate_key_insights()`: Produces actionable recommendations for model training

**Issues Detected**:
- ⚠ No error handling for corrupted images
- ⚠ Sample size hardcoded (50 images) - may miss edge cases
- ⚠ No validation for annotation-image pairing

---

### Module: `prepare_data.py`
**Purpose**: Stratified train/val/test split for OPG Classification dataset

**Function Structure**:
```
prepare_data()
├── Collect images from SOURCE_DIR
├── Stratified split (70/20/10)
│   ├── train_test_split (70/30)
│   └── train_test_split (20/10)
└── Copy files to TARGET_DIR
```

**Dependencies**:
```
External:
├── os, shutil, pathlib
└── sklearn.model_selection.train_test_split

Internal:
└── None
```

**Data Flow**:
```
datasets/xrays/Dental OPG XRAY Dataset/
Dental OPG (Classification)/
    ├── BDC-BDR/
    ├── Caries/
    ├── Fractured Teeth/
    ├── Healthy Teeth/
    ├── Impacted teeth/
    └── Infection/
            │
            ▼
    prepare_data()
            │
            ▼
model1_classification/data/
    ├── train/
    │   ├── BDC-BDR/
    │   ├── Caries/
    │   └── ...
    ├── val/
    └── test/
```

**Critical Operations**:
1. Stratified sampling preserves class distribution
2. Fixed random_state=42 for reproducibility
3. Handles missing class directories gracefully

**Issues Detected**:
- ⚠ No validation for duplicate filenames across classes
- ⚠ No check for corrupted images before copying
- ⚠ Hardcoded paths (not configurable)
- ⚠ No logging of split statistics to file

---

## 3. Dependency Graph

### Python Package Dependencies
```
torch (>=2.0.0)
    └── torchvision (>=0.15.0)
            └── Pillow (>=9.5.0)

ultralytics (>=8.0.0)
    ├── torch
    ├── opencv-python
    └── Pillow

segmentation-models-pytorch (>=0.3.3)
    └── torch

Data Science Stack:
├── numpy (>=1.24.0)
├── pandas (>=2.0.0)
├── scikit-learn (>=1.3.0)
└── opencv-python (>=4.8.0)

Visualization:
├── matplotlib (>=3.7.0)
└── seaborn (>=0.12.0)

Utilities:
├── tqdm (>=4.65.0)
├── split-folders (>=0.5.1)
├── pycocotools (>=2.0.6)
└── supervisely (>=6.73.0)
```

### Module Interdependencies
```
eda_dental_dataset.py
    ├── Uses: numpy, pandas, matplotlib, seaborn, cv2, PIL
    └── Produces: CSV, PNG files

prepare_data.py
    ├── Uses: sklearn, shutil, pathlib
    └── Produces: Organized data directories

requirements.txt
    └── Defines: All external dependencies
```

---

## 4. Data Integrity Analysis

### Annotation Format Compatibility
```
┌─────────────────────────────────────────────────────────────┐
│ Dataset                    │ Format        │ Model Target   │
├────────────────────────────┼───────────────┼────────────────┤
│ Calculus                   │ None          │ ⚠ Needs labels │
│ Cavity Dataset             │ YOLO          │ ✓ Detection    │
│ Teeth Segmentation         │ Supervisely   │ ✓ Segmentation │
│ Bitewing Caries            │ COCO          │ ⚠ Test-only    │
│ OPG Classification         │ Folder labels │ ✓ Class.       │
│ OPG Detection (Original)   │ YOLO          │ ✓ Detection    │
│ OPG Detection (Augmented)  │ YOLO          │ ✓ Detection    │
└─────────────────────────────────────────────────────────────┘
```

### Class Imbalance Report
```
OPG Classification:
┌──────────────────┬───────┬──────────┐
│ Class            │ Count │ % of Max │
├──────────────────┼───────┼──────────┤
│ Healthy Teeth    │  223  │  100%    │
│ Caries           │  119  │   53%    │
│ Impacted teeth   │   87  │   39%    │
│ BDC-BDR          │   52  │   23%    │
│ Infection        │   23  │   10%    │
│ Fractured Teeth  │   13  │    6%    │ ⚠ CRITICAL
└──────────────────┴───────┴──────────┘

Imbalance Ratio: 17.2:1 (Healthy:Fractured)
Recommendation: Use class_weight or focal loss
```

### Split Availability Matrix
```
┌─────────────────────────────┬───────┬─────┬──────┐
│ Dataset                      │ Train │ Val │ Test │
├─────────────────────────────┼───────┼─────┼──────┤
│ Calculus                     │   ✗   │  ✗  │  ✗   │
│ Cavity Dataset               │   ✓   │  ✓  │  ✓   │
│ Teeth Segmentation           │   ✗   │  ✗  │  ✗   │
│ Bitewing Caries              │   ✗   │  ✗  │  ✓   │
│ OPG Classification           │   ✗   │  ✗  │  ✗   │
│ OPG Detection (Original)     │   ✗   │  ✗  │  ✗   │
│ OPG Detection (Augmented)    │   ✓   │  ✓  │  ✓   │
└─────────────────────────────┴───────┴─────┴──────┘

Status: 2/7 datasets ready for training
Action Required: Run prepare_data.py for 5 datasets
```

---

## 5. Model Architecture Plan

### Model 1: Classification (OPG)
```
Input: OPG X-ray images
    │
    ▼
┌─────────────────────────────┐
│ Preprocessing               │
│ - Resize to 224×224         │
│ - Normalize (ImageNet)      │
│ - Augmentation (rotation,   │
│   flip, brightness)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Backbone (Transfer Learning)│
│ - ResNet50 / EfficientNet   │
│ - Pretrained on ImageNet    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Classification Head         │
│ - FC layers                 │
│ - Dropout (0.5)             │
│ - 6 classes output          │
│ - Weighted CrossEntropy     │
└─────────────┬───────────────┘
              │
              ▼
Output: [BDC-BDR, Caries, Fractured, Healthy, Impacted, Infection]

Data Source: model1_classification/data/
Status: ⚠ Needs prepare_data.py execution
```

### Model 2: Detection (Cavity + OPG)
```
Input: Oral photographs / OPG X-rays
    │
    ▼
┌─────────────────────────────┐
│ YOLO Architecture           │
│ - YOLOv8 (ultralytics)      │
│ - Input: 640×640            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Task 1: Cavity Detection    │
│ - Classes: healthy_teeth,   │
│   unhealthy_teeth           │
│ - Data: Cavity Dataset      │
│   (287 train, 93 val)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Task 2: OPG Detection       │
│ - Classes: 6 dental         │
│   conditions                │
│ - Data: OPG Augmented       │
│   (558 train, 23 val)       │
└─────────────┬───────────────┘
              │
              ▼
Output: Bounding boxes + class labels

Data Source: model2_detection/data/
Status: ✓ Pre-split available
```

### Model 3: Segmentation (Teeth)
```
Input: Dental X-ray images
    │
    ▼
┌─────────────────────────────┐
│ U-Net / DeepLabV3+          │
│ - Encoder: ResNet50         │
│ - Decoder: Upsampling       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Per-Tooth Segmentation      │
│ - 32 classes (FDI system)   │
│ - Dice Loss + CE            │
│ - Data: 598 X-rays          │
└─────────────┬───────────────┘
              │
              ▼
Output: Pixel-wise tooth masks (32 classes)

Data Source: model3_segmentation/data/
Status: ⚠ Needs train/val/test split
```

---

## 6. Critical Issues & Recommendations

### 🔴 Critical Issues
1. **Class Imbalance**: OPG Classification has 17:1 ratio
   - Solution: Implement class_weight in loss function
   - Alternative: Use focal loss or SMOTE augmentation

2. **Missing Splits**: 5/7 datasets lack train/val/test splits
   - Solution: Extend prepare_data.py to handle all datasets
   - Priority: Teeth Segmentation (598 images)

3. **Calculus Dataset**: 1,296 images with no annotations
   - Solution: Either label manually or exclude from training
   - Alternative: Use for unsupervised pretraining

4. **Bitewing Caries**: Test-only dataset (100 images)
   - Solution: Use only for final evaluation, not training
   - Risk: Cannot validate model during development

### 🟡 Medium Priority Issues
1. **No data validation pipeline**
   - Add: Image corruption checks
   - Add: Annotation-image pairing validation
   - Add: Class distribution logging

2. **Hardcoded paths in prepare_data.py**
   - Refactor: Use config file (YAML/JSON)
   - Add: Command-line arguments

3. **No model training code present**
   - Create: train_classification.py
   - Create: train_detection.py
   - Create: train_segmentation.py

4. **Missing evaluation metrics**
   - Add: Confusion matrix, F1-score, mAP
   - Add: Per-class performance tracking

### 🟢 Low Priority Enhancements
1. Add data augmentation pipeline
2. Implement cross-validation for small datasets
3. Create unified data loader interface
4. Add TensorBoard logging

---

## 7. Execution Plan

### Phase 1: Data Preparation (Priority: HIGH)
```
1. Run eda_dental_dataset.py
   └── Validate: All datasets accessible, no corruption

2. Extend prepare_data.py
   ├── Add: Teeth Segmentation split
   ├── Add: Calculus split (if labeled)
   └── Add: OPG Detection Original split

3. Validate splits
   └── Check: Class distribution preserved
```

### Phase 2: Model Development (Priority: HIGH)
```
1. Model 1 (Classification)
   ├── Create: train_classification.py
   ├── Implement: Weighted loss for imbalance
   └── Baseline: ResNet50 pretrained

2. Model 2 (Detection)
   ├── Create: train_detection.py
   ├── Use: YOLOv8 from ultralytics
   └── Train: Cavity + OPG datasets separately

3. Model 3 (Segmentation)
   ├── Create: train_segmentation.py
   ├── Use: segmentation-models-pytorch
   └── Architecture: U-Net with ResNet50 encoder
```

### Phase 3: Evaluation (Priority: MEDIUM)
```
1. Create evaluation scripts
   ├── eval_classification.py
   ├── eval_detection.py
   └── eval_segmentation.py

2. Use Bitewing Caries for detection evaluation
3. Generate performance reports
```

---

## 8. File Count Summary

### Current State
```
Python Scripts:        2
├── eda_dental_dataset.py    ✓ Complete
└── prepare_data.py          ✓ Complete (limited scope)

Configuration Files:   1
└── requirements.txt         ✓ Complete

Data Directories:      3
├── model1_classification/data/  ⚠ Empty (needs prepare_data.py)
├── model2_detection/data/       ⚠ Partially populated
└── model3_segmentation/data/    ⚠ Partially populated

Documentation:         2
├── dataset report.txt       ✓ Complete
└── .amazonq/rules/          ✓ Complete

Generated Outputs:     4
├── dataset_summary.csv
├── dataset_overview.png
├── opg_class_imbalance.png
└── opg_classification_*.png
```

### Required Files (Not Present)
```
Training Scripts:      3
├── train_classification.py   ✗ Missing
├── train_detection.py        ✗ Missing
└── train_segmentation.py     ✗ Missing

Evaluation Scripts:    3
├── eval_classification.py    ✗ Missing
├── eval_detection.py         ✗ Missing
└── eval_segmentation.py      ✗ Missing

Utilities:             2
├── data_loader.py            ✗ Missing
└── augmentation.py           ✗ Missing

Configuration:         1
└── config.yaml               ✗ Missing
```

---

## 9. Compliance with Project Rules

### ✓ Followed Rules
1. **File Creation Discipline**: Only 2 Python scripts created (minimal)
2. **Architecture Constraints**: Flat structure, no premature abstraction
3. **Data Integrity**: EDA validates labels, detects imbalance
4. **Dependency Control**: Minimal dependencies in requirements.txt

### ⚠ Violations / Risks
1. **Incomplete Implementation**: No training code (violates "no incomplete implementations")
2. **Missing Validation**: No edge case handling for corrupted images
3. **Fragmentation Risk**: 3 separate model directories (could be consolidated)

### 📋 Recommendations for Compliance
1. Complete training scripts before proceeding
2. Add data validation in prepare_data.py
3. Consider merging model directories into single structure
4. Add error handling for all file operations

---

## 10. Next Immediate Actions

### Action 1: Execute Data Preparation
```bash
cd "d:\college\EDAI\final project"
python model1_classification/prepare_data.py
```
**Expected Output**: model1_classification/data/ populated with train/val/test splits

### Action 2: Validate Data Integrity
```bash
python eda_dental_dataset.py
```
**Expected Output**: Confirm no corrupted images, validate splits

### Action 3: Create Training Script (Classification)
**Priority**: HIGH
**File**: model1_classification/train.py
**Requirements**:
- Load data from model1_classification/data/
- Use weighted CrossEntropyLoss
- Implement early stopping
- Save best model checkpoint

### Action 4: Address Class Imbalance
**Method**: Compute class weights
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
```

---

## Summary Statistics

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT HEALTH REPORT                     │
├─────────────────────────────────────────────────────────────┤
│ Total Datasets:              7                               │
│ Total Images:                ~3,764                          │
│ Datasets Ready for Training: 2/7 (29%)                       │
│ Python Scripts:              2                               │
│ Missing Critical Files:      9                               │
│ Class Imbalance Issues:      1 (OPG Classification)          │
│ Test-Only Datasets:          1 (Bitewing Caries)             │
│ Annotation Formats:          4 (YOLO, COCO, Supervisely, Folder) │
│ Model Tasks:                 3 (Classification, Detection, Segmentation) │
├─────────────────────────────────────────────────────────────┤
│ Overall Status:              🟡 INCOMPLETE                   │
│ Readiness for Training:      30%                             │
│ Code Quality:                ✓ GOOD (minimal, focused)       │
│ Data Quality:                ⚠ NEEDS VALIDATION              │
└─────────────────────────────────────────────────────────────┘
```

**Conclusion**: Project has solid data foundation and analysis tools, but lacks training implementation. Priority: Complete data preparation and create training scripts.
