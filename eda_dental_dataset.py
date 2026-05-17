import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

BASE_PATH = r"d:\college\EDAI\final project\datasets"

class DentalDatasetEDA:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = {}
        self.all_stats = []
        
    def count_files(self, directory, extensions):
        """Count files with specific extensions"""
        if not os.path.exists(directory):
            return 0
        try:
            return len([f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)])
        except:
            return 0
    
    def get_image_dimensions(self, directory, extensions, sample_size=50):
        """Get image dimensions statistics"""
        sizes = []
        if not os.path.exists(directory):
            return None
        try:
            files = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
            for img_file in files[:min(sample_size, len(files))]:
                try:
                    img = Image.open(os.path.join(directory, img_file))
                    sizes.append(img.size)
                except:
                    pass
        except:
            pass
        
        if sizes:
            sizes_array = np.array(sizes)
            return {
                'avg_width': int(np.mean(sizes_array[:, 0])),
                'avg_height': int(np.mean(sizes_array[:, 1])),
                'min_width': int(np.min(sizes_array[:, 0])),
                'max_width': int(np.max(sizes_array[:, 0])),
                'min_height': int(np.min(sizes_array[:, 1])),
                'max_height': int(np.max(sizes_array[:, 1]))
            }
        return None
    
    def analyze_yolo_labels(self, directory, sample_size=100):
        """Analyze YOLO format labels"""
        if not os.path.exists(directory):
            return None
        
        try:
            label_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
            class_counts = Counter()
            bbox_counts = []
            
            for label_file in label_files[:min(sample_size, len(label_files))]:
                try:
                    with open(os.path.join(directory, label_file), 'r') as f:
                        lines = f.readlines()
                        bbox_counts.append(len(lines))
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                except:
                    pass
            
            return {
                'total_labels': len(label_files),
                'avg_bboxes_per_image': np.mean(bbox_counts) if bbox_counts else 0,
                'max_bboxes': max(bbox_counts) if bbox_counts else 0,
                'min_bboxes': min(bbox_counts) if bbox_counts else 0,
                'class_distribution': dict(class_counts)
            }
        except:
            return None
    
    def analyze_calculus(self):
        """Analyze Calculus dataset"""
        path = os.path.join(self.base_path, "normal photographs", "Calculus")
        images = self.count_files(path, ['.jpg', '.JPG'])
        
        stats = {
            'dataset': 'Calculus',
            'modality': 'Normal Oral Photograph',
            'task': 'Image Classification (Calculus Detection)',
            'total_images': images,
            'annotations': 'None',
            'train_val_test_split': 'No',
            'image_formats': ['JPG'],
            'classes': 'Single-class (Calculus)',
            'notes': 'No annotation files. Suitable for unsupervised/self-supervised learning.'
        }
        
        dims = self.get_image_dimensions(path, ['.jpg', '.JPG'])
        if dims:
            stats.update(dims)
        
        return stats
    
    def analyze_cavity_dataset(self):
        """Analyze Cavity Dataset (YOLO format)"""
        path = os.path.join(self.base_path, "normal photographs", "Cavity Dataset")
        
        stats = {
            'dataset': 'Cavity Dataset',
            'modality': 'Normal Oral Photograph',
            'task': 'Object Detection (YOLO)',
            'annotation_format': 'YOLO .txt',
            'classes': ['healthy_teeth', 'unhealthy_teeth'],
            'num_classes': 2,
            'license': 'CC BY 4.0',
            'source': 'Roboflow - CAV-TEE Research Team'
        }
        
        splits = {}
        total_images = 0
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(path, split)
            images = self.count_files(split_path, ['.jpg', '.png', '.JPG', '.PNG'])
            splits[split] = images
            total_images += images
        
        stats['splits'] = splits
        stats['total_images'] = total_images
        
        # Analyze YOLO labels in train split
        train_path = os.path.join(path, 'train')
        label_stats = self.analyze_yolo_labels(train_path)
        if label_stats:
            stats.update(label_stats)
        
        # Get image dimensions from train
        dims = self.get_image_dimensions(train_path, ['.jpg', '.png', '.JPG', '.PNG'])
        if dims:
            stats.update(dims)
        
        return stats
    
    def analyze_teeth_segmentation(self):
        """Analyze Teeth Segmentation dataset (DatasetNinja)"""
        path = os.path.join(self.base_path, "teeth-segmentation-on-dental-x-ray-images-DatasetNinja")
        
        stats = {
            'dataset': 'Teeth Segmentation (DatasetNinja)',
            'modality': 'Dental X-ray',
            'task': 'Semantic Segmentation (Per-tooth Instance Masks)',
            'annotation_format': 'Supervisely JSON with bitmap masks',
            'num_classes': 32,
            'class_system': 'FDI Tooth Numbering (Teeth 1-32)',
            'train_val_test_split': 'No'
        }
        
        img_path = os.path.join(path, 'ds', 'img')
        ann_path = os.path.join(path, 'ds', 'ann')
        
        images = self.count_files(img_path, ['.jpg', '.JPG'])
        annotations = self.count_files(ann_path, ['.json'])
        
        stats['total_images'] = images
        stats['total_annotations'] = annotations
        
        # Analyze sample annotation
        if os.path.exists(ann_path):
            try:
                ann_files = [f for f in os.listdir(ann_path) if f.endswith('.json')]
                if ann_files:
                    with open(os.path.join(ann_path, ann_files[0]), 'r') as f:
                        sample_ann = json.load(f)
                        if 'objects' in sample_ann:
                            stats['avg_objects_per_image'] = len(sample_ann['objects'])
            except:
                pass
        
        # Load meta.json
        meta_path = os.path.join(path, 'meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    if 'classes' in meta:
                        stats['classes_info'] = len(meta['classes'])
            except:
                pass
        
        # Get image dimensions
        dims = self.get_image_dimensions(img_path, ['.jpg', '.JPG'])
        if dims:
            stats.update(dims)
        
        stats['notes'] = 'Most richly annotated dataset with per-tooth bitmap masks for all 32 teeth.'
        
        return stats
    
    def analyze_bitewing_caries(self):
        """Analyze Dental Caries in Bitewing Radiographs"""
        path = os.path.join(self.base_path, "xrays", "Dental caries in bitewing radiographs")
        
        stats = {
            'dataset': 'Dental Caries in Bitewing Radiographs',
            'modality': 'Bitewing Dental X-ray',
            'task': 'Object Detection (Carious Lesion Detection)',
            'annotation_format': 'COCO JSON',
            'image_format': 'PNG',
            'image_resolution': '1068×847 pixels',
            'source': 'Czech Technical University / Charles University Prague',
            'dataset_name': 'D0',
            'train_val_test_split': 'Test-only (no training split)',
            'num_annotators': 8,
            'annotator_types': {
                'experts': '5 (E0-E4, >15 years experience)',
                'novices': '3 (N1-N3, <5 years experience)'
            }
        }
        
        img_path = os.path.join(path, 'images')
        images = self.count_files(img_path, ['.png', '.PNG'])
        stats['total_images'] = images
        
        # Analyze COCO annotations
        ann_file = os.path.join(path, 'test_annotations_anonymized.json')
        if os.path.exists(ann_file):
            try:
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                    stats['total_annotations'] = len(coco_data.get('annotations', []))
                    stats['num_categories'] = len(coco_data.get('categories', []))
                    
                    img_bbox_count = Counter()
                    for ann in coco_data.get('annotations', []):
                        img_bbox_count[ann['image_id']] += 1
                    
                    if img_bbox_count:
                        stats['avg_bboxes_per_image'] = np.mean(list(img_bbox_count.values()))
                        stats['max_bboxes_in_image'] = max(img_bbox_count.values())
            except:
                pass
        
        # Get image dimensions
        dims = self.get_image_dimensions(img_path, ['.png', '.PNG'])
        if dims:
            stats.update(dims)
        
        stats['notes'] = 'Test-only dataset. Cannot be used for training, only evaluation. Published in Springer Clinical Oral Investigations.'
        
        return stats
    
    def analyze_opg_classification(self):
        """Analyze OPG Classification dataset"""
        path = os.path.join(self.base_path, "xrays", "Dental OPG XRAY Dataset", "Dental OPG (Classification)")
        
        stats = {
            'dataset': 'Dental OPG (Classification)',
            'modality': 'OPG (Orthopantomogram) X-ray',
            'task': 'Multi-class Image Classification',
            'image_format': 'JPG',
            'train_val_test_split': 'No',
            'class_imbalance': 'Significant'
        }
        
        class_counts = {}
        total_images = 0
        
        if os.path.exists(path):
            for class_name in os.listdir(path):
                class_path = os.path.join(path, class_name)
                if os.path.isdir(class_path):
                    images = self.count_files(class_path, ['.jpg', '.JPG'])
                    class_counts[class_name] = images
                    total_images += images
        
        stats['classes'] = class_counts
        stats['total_images'] = total_images
        stats['num_classes'] = len(class_counts)
        
        if class_counts:
            max_class = max(class_counts.values())
            min_class = min(class_counts.values())
            stats['imbalance_ratio'] = f"{max_class}:{min_class} ({max_class/min_class:.1f}x)"
        
        # Get image dimensions from first class
        if class_counts:
            first_class = list(class_counts.keys())[0]
            first_class_path = os.path.join(path, first_class)
            dims = self.get_image_dimensions(first_class_path, ['.jpg', '.JPG'])
            if dims:
                stats.update(dims)
        
        return stats
    
    def analyze_opg_detection_original(self):
        """Analyze OPG Object Detection (Original)"""
        path = os.path.join(self.base_path, "xrays", "Dental OPG XRAY Dataset", "Dental OPG (Object Detection)", "Original Dataset")
        
        stats = {
            'dataset': 'Dental OPG (Object Detection - Original)',
            'modality': 'OPG X-ray',
            'task': 'Object Detection (YOLO)',
            'annotation_format': 'YOLO .txt',
            'num_classes': 6,
            'classes': ['BDC-BDR', 'Caries', 'Fractured Teeth', 'Healthy Teeth', 'Impacted teeth', 'Infection'],
            'train_val_test_split': 'No',
            'augmentation': 'No'
        }
        
        if os.path.exists(path):
            images = self.count_files(path, ['.jpg', '.JPG'])
            labels = self.count_files(path, ['.txt'])
            stats['total_images'] = images
            stats['total_labels'] = labels
            
            # Analyze YOLO labels
            label_stats = self.analyze_yolo_labels(path)
            if label_stats:
                stats.update(label_stats)
            
            # Get image dimensions
            dims = self.get_image_dimensions(path, ['.jpg', '.JPG'])
            if dims:
                stats.update(dims)
        
        return stats
    
    def analyze_opg_detection_augmented(self):
        """Analyze OPG Object Detection (Augmented)"""
        path = os.path.join(self.base_path, "xrays", "Dental OPG XRAY Dataset", "Dental OPG (Object Detection)", "Augmented Dataset")
        
        stats = {
            'dataset': 'Dental OPG (Object Detection - Augmented)',
            'modality': 'OPG X-ray',
            'task': 'Object Detection (YOLO)',
            'annotation_format': 'YOLO .txt',
            'num_classes': 6,
            'classes': ['BDC-BDR', 'Caries', 'Fractured Teeth', 'Healthy Teeth', 'Impacted teeth', 'Infection'],
            'train_val_test_split': 'Yes',
            'augmentation': 'Yes (2.6x augmentation factor)',
            'augmentation_note': 'Training set augmented from ~231 to 558 images'
        }
        
        splits = {}
        total_images = 0
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(path, split)
            images = self.count_files(split_path, ['.jpg', '.JPG'])
            splits[split] = images
            total_images += images
        
        stats['splits'] = splits
        stats['total_images'] = total_images
        
        # Analyze YOLO labels in train split
        train_path = os.path.join(path, 'train')
        label_stats = self.analyze_yolo_labels(train_path)
        if label_stats:
            stats.update(label_stats)
        
        # Get image dimensions from train
        dims = self.get_image_dimensions(train_path, ['.jpg', '.JPG'])
        if dims:
            stats.update(dims)
        
        return stats
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*100)
        print("DENTAL AI DATASET - COMPREHENSIVE EDA REPORT")
        print("="*100 + "\n")
        
        datasets = {
            'Calculus': self.analyze_calculus(),
            'Cavity Dataset': self.analyze_cavity_dataset(),
            'Teeth Segmentation': self.analyze_teeth_segmentation(),
            'Bitewing Caries': self.analyze_bitewing_caries(),
            'OPG Classification': self.analyze_opg_classification(),
            'OPG Detection (Original)': self.analyze_opg_detection_original(),
            'OPG Detection (Augmented)': self.analyze_opg_detection_augmented()
        }
        
        self.results = datasets
        
        for dataset_name, stats in datasets.items():
            print(f"\n{'─'*100}")
            print(f"DATASET: {dataset_name}")
            print(f"{'─'*100}")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}: {', '.join(map(str, value))}")
                else:
                    print(f"  {key}: {value}")
        
        return datasets
    
    def create_summary_dataframe(self):
        """Create summary dataframe"""
        summary_data = []
        
        for dataset_name, stats in self.results.items():
            row = {
                'Dataset': dataset_name,
                'Modality': stats.get('modality', 'N/A'),
                'Task': stats.get('task', 'N/A'),
                'Total Images': stats.get('total_images', 'N/A'),
                'Annotation Format': stats.get('annotation_format', stats.get('annotations', 'N/A')),
                'Has Train/Val/Test Split': stats.get('train_val_test_split', 'N/A'),
                'Classes/Categories': stats.get('num_classes', stats.get('classes', 'N/A'))
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)
        print(df.to_string(index=False))
        
        return df
    
    def create_visualizations(self):
        """Create visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        datasets_list = list(self.results.keys())
        image_counts = [self.results[d].get('total_images', 0) for d in datasets_list]
        
        # Image count by dataset
        ax = axes[0, 0]
        bars = ax.barh(datasets_list, image_counts, color='steelblue')
        ax.set_xlabel('Number of Images', fontsize=11, fontweight='bold')
        ax.set_title('Total Images per Dataset', fontsize=12, fontweight='bold')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                   ha='left', va='center', fontsize=10)
        
        # Modality distribution
        ax = axes[0, 1]
        modalities = [self.results[d].get('modality', 'Unknown') for d in datasets_list]
        modality_counts = Counter(modalities)
        ax.pie(modality_counts.values(), labels=modality_counts.keys(), autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff', '#99ff99'])
        ax.set_title('Imaging Modality Distribution', fontsize=12, fontweight='bold')
        
        # Task distribution
        ax = axes[1, 0]
        tasks = [self.results[d].get('task', 'Unknown') for d in datasets_list]
        task_counts = Counter(tasks)
        ax.barh(list(task_counts.keys()), list(task_counts.values()), color='coral')
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Task Type Distribution', fontsize=12, fontweight='bold')
        
        # Train/Val/Test split availability
        ax = axes[1, 1]
        split_status = [self.results[d].get('train_val_test_split', 'Unknown') for d in datasets_list]
        split_counts = Counter(split_status)
        colors_split = ['#90EE90' if 'Yes' in str(k) else '#FFB6C6' for k in split_counts.keys()]
        ax.bar(range(len(split_counts)), list(split_counts.values()), color=colors_split)
        ax.set_xticks(range(len(split_counts)))
        ax.set_xticklabels(split_counts.keys(), rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title('Train/Val/Test Split Availability', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_path, '..', 'dataset_overview.png'), dpi=300, bbox_inches='tight')
        print("\n✓ Saved: dataset_overview.png")
        
        # OPG Classification class imbalance
        opg_classes = self.results['OPG Classification'].get('classes', {})
        if opg_classes:
            fig, ax = plt.subplots(figsize=(12, 6))
            classes = list(opg_classes.keys())
            counts = list(opg_classes.values())
            colors_imbalance = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(classes)))
            bars = ax.bar(classes, counts, color=colors_imbalance)
            ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
            ax.set_title('OPG Classification - Class Imbalance', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_path, '..', 'opg_class_imbalance.png'), dpi=300, bbox_inches='tight')
            print("✓ Saved: opg_class_imbalance.png")
        
        # OPG Detection augmentation effect
        opg_aug = self.results['OPG Detection (Augmented)'].get('splits', {})
        if opg_aug:
            fig, ax = plt.subplots(figsize=(10, 6))
            splits = list(opg_aug.keys())
            counts = list(opg_aug.values())
            colors_aug = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(splits, counts, color=colors_aug)
            ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
            ax.set_title('OPG Detection (Augmented) - Train/Val/Test Split', fontsize=12, fontweight='bold')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_path, '..', 'opg_augmentation_split.png'), dpi=300, bbox_inches='tight')
            print("✓ Saved: opg_augmentation_split.png")
        
        # Cavity Dataset splits
        cavity_splits = self.results['Cavity Dataset'].get('splits', {})
        if cavity_splits:
            fig, ax = plt.subplots(figsize=(10, 6))
            splits = list(cavity_splits.keys())
            counts = list(cavity_splits.values())
            colors_cavity = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(splits, counts, color=colors_cavity)
            ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
            ax.set_title('Cavity Dataset - Train/Val/Test Split', fontsize=12, fontweight='bold')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.base_path, '..', 'cavity_dataset_split.png'), dpi=300, bbox_inches='tight')
            print("✓ Saved: cavity_dataset_split.png")
    
    def generate_key_insights(self):
        """Generate key insights and recommendations"""
        print("\n" + "="*100)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*100 + "\n")
        
        insights = [
            ("Dataset Diversity", 
             "✓ 7 distinct datasets covering 2 imaging modalities (photographs & X-rays)\n"
             "  ✓ 3 task types: Classification, Object Detection, Semantic Segmentation\n"
             "  ✓ Total ~3,764 images across all datasets"),
            
            ("Annotation Quality",
             "✓ Teeth Segmentation: Most richly annotated (598 images with per-tooth masks)\n"
             "  ✓ Cavity Dataset: Well-structured YOLO format with proper splits\n"
             "  ✓ Bitewing Caries: Multi-annotator (8 experts/novices) for robustness"),
            
            ("Data Imbalance Issues",
             "⚠ OPG Classification: Severe class imbalance (223 Healthy vs 13 Fractured Teeth)\n"
             "  ⚠ Recommendation: Use weighted loss functions or SMOTE for balancing"),
            
            ("Train/Val/Test Splits",
             "✓ Only 2 datasets have proper splits (Cavity Dataset, OPG Augmented)\n"
             "  ⚠ 5 datasets need manual splitting before training\n"
             "  ⚠ Bitewing Caries is test-only (evaluation only)"),
            
            ("Data Augmentation",
             "✓ OPG Detection: Augmented dataset available (2.6x factor)\n"
             "  ⚠ Other datasets may benefit from augmentation (rotation, flip, brightness)"),
            
            ("Annotation Formats",
             "✓ YOLO format: 3 datasets (standardized for detection)\n"
             "  ✓ COCO JSON: 1 dataset (flexible, multi-annotator support)\n"
             "  ✓ Supervisely JSON: 1 dataset (rich segmentation masks)"),
            
            ("Recommendations for Model Development",
             "1. Start with Cavity Dataset (balanced, pre-split)\n"
             "  2. Use OPG Augmented for detection tasks\n"
             "  3. Address class imbalance in OPG Classification\n"
             "  4. Leverage Teeth Segmentation for transfer learning\n"
             "  5. Use Bitewing Caries for final evaluation only")
        ]
        
        for title, content in insights:
            print(f"\n{title}:")
            print(f"  {content}")
        
        print("\n" + "="*100 + "\n")

def main():
    eda = DentalDatasetEDA(BASE_PATH)
    
    print("\n" + "="*100)
    print("STARTING COMPREHENSIVE EDA ANALYSIS...")
    print("="*100)
    
    datasets = eda.generate_summary_report()
    summary_df = eda.create_summary_dataframe()
    
    print("\n" + "="*100)
    print("GENERATING VISUALIZATIONS...")
    print("="*100)
    eda.create_visualizations()
    
    eda.generate_key_insights()
    
    output_path = os.path.join(BASE_PATH, '..', 'dataset_summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved: dataset_summary.csv")
    
    print("\n" + "="*100)
    print("EDA COMPLETE!")
    print("="*100)
    print("\nGenerated files:")
    print("  - dataset_overview.png")
    print("  - opg_class_imbalance.png")
    print("  - opg_augmentation_split.png")
    print("  - cavity_dataset_split.png")
    print("  - dataset_summary.csv")

if __name__ == "__main__":
    main()
