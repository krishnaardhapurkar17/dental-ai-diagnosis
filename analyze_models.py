import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r'd:\college\EDAI\final project'
RESEARCH_OUTPUT_DIR = os.path.join(BASE_DIR, 'research_analysis')

# Create output directory
os.makedirs(RESEARCH_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations'), exist_ok=True)
os.makedirs(os.path.join(RESEARCH_OUTPUT_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(RESEARCH_OUTPUT_DIR, 'reports'), exist_ok=True)
os.makedirs(os.path.join(RESEARCH_OUTPUT_DIR, 'data'), exist_ok=True)

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_size(path):
    """Get model file size in MB"""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def save_table(df, task_name, filename, format_type='csv'):
    """Save table in multiple formats"""
    base_path = os.path.join(RESEARCH_OUTPUT_DIR, 'tables', f"{task_name}_{filename}")
    
    if format_type == 'csv':
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    elif format_type == 'html':
        html_path = f"{base_path}.html"
        df.to_html(html_path, index=False, border=0, justify='center')
        return html_path
    elif format_type == 'latex':
        latex_path = f"{base_path}.tex"
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False))
        return latex_path

def create_comparison_chart(df, x_col, y_col, title, filename, hue_col=None):
    """Create comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if hue_col:
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="husl")
    else:
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax, palette="husl")
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    path = os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations', f"{filename}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path

def create_line_plot(df, x_col, y_col, title, filename, hue_col=None):
    """Create line plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if hue_col:
        for category in df[hue_col].unique():
            subset = df[df[hue_col] == category]
            ax.plot(subset[x_col], subset[y_col], marker='o', label=category, linewidth=2, markersize=8)
        ax.legend(fontsize=10)
    else:
        ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=8, color='#1f77b4')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations', f"{filename}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path

def create_heatmap(data, title, filename, annot=True):
    """Create heatmap for correlation/comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, annot=annot, fmt='.3f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    path = os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations', f"{filename}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path

def create_efficiency_plot(df, task_name):
    """Create model efficiency plot (accuracy vs model size)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out rows with NaN values for size and main metric
    metric_col = [col for col in df.columns if col not in ['model', 'type', 'size_mb']][0]
    df_clean = df.dropna(subset=[metric_col, 'size_mb'])
    
    if len(df_clean) > 0:
        scatter = ax.scatter(df_clean['size_mb'], df_clean[metric_col], 
                            s=200, alpha=0.6, c=range(len(df_clean)), cmap='viridis')
        
        for idx, row in df_clean.iterrows():
            ax.annotate(row['model'], (row['size_mb'], row[metric_col]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_name}: Performance vs Model Size Trade-off', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations', f"{task_name}_efficiency.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_classification():
    """Analyze classification models - comparative evaluation only"""
    print("\n" + "="*70)
    print("CLASSIFICATION MODELS ANALYSIS")
    print("="*70)
    
    # Load comparative results (all models with complete metrics)
    comp_results = pd.read_csv(os.path.join(BASE_DIR, 'comparative_models', 'model1_classification', 'comparative_results.csv'))
    
    results = []
    for _, row in comp_results.iterrows():
        model_name = row['model']
        path = os.path.join(BASE_DIR, 'comparative_models', 'model1_classification', 
                           f"{model_name.lower().replace('-', '')}_best.pth")
        results.append({
            'model': model_name,
            'accuracy': float(row['accuracy']),
            'precision': float(row['precision']),
            'recall': float(row['recall']),
            'f1_score': float(row['f1_score']),
            'size_mb': get_model_size(path)
        })
    
    df = pd.DataFrame(results)
    
    # Save tables
    save_table(df, 'classification', 'performance_metrics', 'csv')
    save_table(df, 'classification', 'performance_metrics', 'html')
    save_table(df, 'classification', 'performance_metrics', 'latex')
    df.to_csv(os.path.join(RESEARCH_OUTPUT_DIR, 'data', 'classification_data.csv'), index=False)
    
    print("\n--- Performance Metrics ---")
    print(df.to_string(index=False))
    
    # Visualizations
    create_comparison_chart(df, 'model', 'accuracy', 
                          'Classification Model Accuracy Comparison',
                          'classification_accuracy_comparison')
    
    df_metrics_melted = df.melt(id_vars='model', value_vars=['accuracy', 'precision', 'recall', 'f1_score'], 
                                var_name='metric', value_name='score')
    create_comparison_chart(df_metrics_melted, 'model', 'score', 
                          'Classification Models - All Metrics',
                          'classification_all_metrics', 'metric')
    
    create_efficiency_plot(df, 'Classification')
    
    # Statistical summary
    print("\n--- Statistical Summary ---")
    print(f"Mean Accuracy: {df['accuracy'].mean():.2f}%")
    print(f"Std Accuracy: {df['accuracy'].std():.2f}%")
    print(f"Best Accuracy: {df['accuracy'].max():.2f}%")
    print(f"Worst Accuracy: {df['accuracy'].min():.2f}%")
    print(f"Mean Precision: {df['precision'].mean():.4f}")
    print(f"Mean Recall: {df['recall'].mean():.4f}")
    print(f"Mean F1-Score: {df['f1_score'].mean():.4f}")
    
    best = df.loc[df['accuracy'].idxmax()]
    print(f"\n✓ BEST MODEL: {best['model']} (Accuracy: {best['accuracy']:.2f}%)")
    return best, df

def analyze_detection():
    """Analyze detection models - comprehensive comparison"""
    print("\n" + "="*70)
    print("DETECTION MODELS ANALYSIS")
    print("="*70)
    
    # Load comparative results
    comp_results = pd.read_csv(os.path.join(BASE_DIR, 'comparative_models', 'model2_detection', 'comparative_results.csv'))
    
    # Cavity Detection Analysis
    print("\n--- Cavity Detection Models ---")
    cavity_results = []
    cavity_data = comp_results[comp_results['model'].str.contains('Cavity', na=False)]
    
    for _, row in cavity_data.iterrows():
        cavity_results.append({
            'model': row['model'],
            'mAP50': float(row['mAP50']),
            'mAP50-95': float(row['mAP50-95']),
            'precision': float(row['precision']),
            'recall': float(row['recall'])
        })
    
    cavity_df = pd.DataFrame(cavity_results)
    print(cavity_df.to_string(index=False))
    
    # Save cavity tables
    save_table(cavity_df, 'detection_cavity', 'performance_metrics', 'csv')
    save_table(cavity_df, 'detection_cavity', 'performance_metrics', 'html')
    cavity_df.to_csv(os.path.join(RESEARCH_OUTPUT_DIR, 'data', 'cavity_detection_data.csv'), index=False)
    
    # OPG Detection Analysis
    print("\n--- OPG Detection Models ---")
    opg_results = []
    opg_data = comp_results[comp_results['model'].str.contains('OPG', na=False)]
    
    for _, row in opg_data.iterrows():
        opg_results.append({
            'model': row['model'],
            'mAP50': float(row['mAP50']),
            'mAP50-95': float(row['mAP50-95']),
            'precision': float(row['precision']),
            'recall': float(row['recall'])
        })
    
    opg_df = pd.DataFrame(opg_results)
    print(opg_df.to_string(index=False))
    
    # Save OPG tables
    save_table(opg_df, 'detection_opg', 'performance_metrics', 'csv')
    save_table(opg_df, 'detection_opg', 'performance_metrics', 'html')
    opg_df.to_csv(os.path.join(RESEARCH_OUTPUT_DIR, 'data', 'opg_detection_data.csv'), index=False)
    
    # Visualizations
    create_comparison_chart(cavity_df, 'model', 'mAP50', 
                          'Cavity Detection - mAP50 Comparison',
                          'cavity_detection_map50')
    
    create_comparison_chart(opg_df, 'model', 'mAP50', 
                          'OPG Detection - mAP50 Comparison',
                          'opg_detection_map50')
    
    # Combined detection metrics
    cavity_df_melted = cavity_df.melt(id_vars='model', value_vars=['mAP50', 'mAP50-95', 'precision', 'recall'],
                                       var_name='metric', value_name='score')
    create_comparison_chart(cavity_df_melted, 'model', 'score',
                          'Cavity Detection - All Metrics',
                          'cavity_detection_all_metrics', 'metric')
    
    opg_df_melted = opg_df.melt(id_vars='model', value_vars=['mAP50', 'mAP50-95', 'precision', 'recall'],
                                 var_name='metric', value_name='score')
    create_comparison_chart(opg_df_melted, 'model', 'score',
                          'OPG Detection - All Metrics',
                          'opg_detection_all_metrics', 'metric')
    
    # Statistical summary
    print("\n--- Cavity Detection - Statistical Summary ---")
    print(f"Mean mAP50: {cavity_df['mAP50'].mean():.4f}")
    print(f"Std mAP50: {cavity_df['mAP50'].std():.4f}")
    print(f"Best mAP50: {cavity_df['mAP50'].max():.4f}")
    print(f"Mean Precision: {cavity_df['precision'].mean():.4f}")
    print(f"Mean Recall: {cavity_df['recall'].mean():.4f}")
    
    print("\n--- OPG Detection - Statistical Summary ---")
    print(f"Mean mAP50: {opg_df['mAP50'].mean():.4f}")
    print(f"Std mAP50: {opg_df['mAP50'].std():.4f}")
    print(f"Best mAP50: {opg_df['mAP50'].max():.4f}")
    print(f"Mean Precision: {opg_df['precision'].mean():.4f}")
    print(f"Mean Recall: {opg_df['recall'].mean():.4f}")
    
    best_cavity = cavity_df.loc[cavity_df['mAP50'].idxmax()]
    best_opg = opg_df.loc[opg_df['mAP50'].idxmax()]
    
    print(f"\n✓ BEST CAVITY MODEL: {best_cavity['model']} (mAP50: {best_cavity['mAP50']:.4f})")
    print(f"✓ BEST OPG MODEL: {best_opg['model']} (mAP50: {best_opg['mAP50']:.4f})")
    
    return best_cavity, best_opg, cavity_df, opg_df

def analyze_segmentation():
    """Analyze segmentation models (base + 2 comparative)"""
    print("\n" + "="*70)
    print("SEGMENTATION MODELS ANALYSIS")
    print("="*70)
    
    # Base model
    base_history = pd.read_csv(os.path.join(BASE_DIR, 'model3_segmentation', 'checkpoints', 'training_history.csv'))
    base_iou = base_history['val_iou'].max()
    
    # Comparative models
    comp_results = pd.read_csv(os.path.join(BASE_DIR, 'comparative_models', 'model3_segmentation', 'comparative_results.csv'))
    
    results = []
    results.append({
        'model': 'U-Net (Base)',
        'mean_iou': base_iou,
        'size_mb': get_model_size(os.path.join(BASE_DIR, 'model3_segmentation', 'checkpoints', 'best_model.pth')),
        'type': 'Base'
    })
    
    for _, row in comp_results.iterrows():
        results.append({
            'model': row['model'],
            'mean_iou': row['mean_iou'],
            'size_mb': 0,
            'type': 'Comparative'
        })
    
    df = pd.DataFrame(results)
    df['mean_iou'] = pd.to_numeric(df['mean_iou'], errors='coerce')
    
    # Save tables
    save_table(df, 'segmentation', 'performance_metrics', 'csv')
    save_table(df, 'segmentation', 'performance_metrics', 'html')
    save_table(df, 'segmentation', 'performance_metrics', 'latex')
    df.to_csv(os.path.join(RESEARCH_OUTPUT_DIR, 'data', 'segmentation_data.csv'), index=False)
    
    print("\n--- Performance Metrics ---")
    print(df.to_string(index=False))
    
    # Visualizations
    create_comparison_chart(df, 'model', 'mean_iou', 
                          'Segmentation Models - Mean IoU Comparison',
                          'segmentation_iou_comparison')
    
    create_efficiency_plot(df, 'Segmentation')
    
    # Statistical summary
    print("\n--- Statistical Summary ---")
    print(f"Mean IoU: {df['mean_iou'].mean():.4f}")
    print(f"Std IoU: {df['mean_iou'].std():.4f}")
    print(f"Best IoU: {df['mean_iou'].max():.4f}")
    print(f"Worst IoU: {df['mean_iou'].min():.4f}")
    
    best = df.loc[df['mean_iou'].idxmax()]
    print(f"\n✓ BEST MODEL: {best['model']} (IoU: {best['mean_iou']:.4f})")
    return best, df

def analyze_clinical():
    """Analyze clinical classification models (base + 2 comparative)"""
    print("\n" + "="*70)
    print("CLINICAL CLASSIFICATION MODELS ANALYSIS")
    print("="*70)
    
    # Base model
    base_path = os.path.join(BASE_DIR, 'model4_clinical_classification', 'best_clinical_photo_model.pth')
    
    # Comparative models
    comp_results = pd.read_csv(os.path.join(BASE_DIR, 'comparative_models', 'model4_clinical', 'comparative_results.csv'))
    
    results = []
    # Add base model only if path exists and has metrics available in comparative results
    results.append({
        'model': 'ResNet-50 (Base)',
        'accuracy': np.nan,
        'precision': np.nan,
        'recall': np.nan,
        'f1_score': np.nan,
        'size_mb': get_model_size(base_path),
        'type': 'Base'
    })
    
    for _, row in comp_results.iterrows():
        results.append({
            'model': row['model'],
            'accuracy': row['accuracy'],
            'precision': row.get('precision', np.nan),
            'recall': row.get('recall', np.nan),
            'f1_score': row['f1_score'],
            'size_mb': 0,
            'type': 'Comparative'
        })
    
    df = pd.DataFrame(results)
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['precision'] = pd.to_numeric(df['precision'], errors='coerce')
    df['recall'] = pd.to_numeric(df['recall'], errors='coerce')
    df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')
    
    # Save tables
    save_table(df, 'clinical', 'performance_metrics', 'csv')
    save_table(df, 'clinical', 'performance_metrics', 'html')
    df.to_csv(os.path.join(RESEARCH_OUTPUT_DIR, 'data', 'clinical_data.csv'), index=False)
    
    print("\n--- Performance Metrics ---")
    print(df.to_string(index=False))
    
    # Visualizations - only for models with accuracy data
    df_clean = df.dropna(subset=['accuracy'])
    if len(df_clean) > 0:
        create_comparison_chart(df_clean, 'model', 'accuracy', 
                              'Clinical Classification - Accuracy Comparison',
                              'clinical_accuracy_comparison')
        
        df_metrics = df_clean[['model', 'accuracy', 'f1_score']].dropna()
        if len(df_metrics) > 0:
            df_metrics_melted = df_metrics.melt(id_vars='model', var_name='metric', value_name='score')
            create_comparison_chart(df_metrics_melted, 'model', 'score',
                                  'Clinical Classification - All Metrics',
                                  'clinical_all_metrics', 'metric')
        
        create_efficiency_plot(df, 'Clinical Classification')
    
    # Statistical summary
    print("\n--- Statistical Summary ---")
    if df['accuracy'].notna().sum() > 0:
        print(f"Mean Accuracy: {df['accuracy'].mean():.2f}%")
        print(f"Std Accuracy: {df['accuracy'].std():.2f}%")
        print(f"Best Accuracy: {df['accuracy'].max():.2f}%")
    
    if df['f1_score'].notna().sum() > 0:
        print(f"Mean F1-Score: {df['f1_score'].mean():.4f}")
    
    best = df.loc[df['accuracy'].idxmax()]
    if pd.notna(best['accuracy']):
        print(f"\n✓ BEST MODEL: {best['model']} (Accuracy: {best['accuracy']:.2f}%, F1: {best['f1_score']:.4f})")
    else:
        print(f"\n✓ BEST MODEL: {best['model']}")
    return best, df

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_research_report(best_class, df_class, best_cavity, best_opg, df_cavity, df_opg,
                            best_seg, df_seg, best_clinical, df_clinical):
    """Generate comprehensive research report"""
    report_path = os.path.join(RESEARCH_OUTPUT_DIR, 'reports', 'research_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
        f.write("Dental Disease Detection and Segmentation Project\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"This report presents a comprehensive analysis of four complementary models for\n")
        f.write(f"dental disease detection, classification, and segmentation.\n\n")
        
        # Classification Section
        f.write("1. CLASSIFICATION MODELS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {len(df_class)}\n")
        f.write(f"Best Model: {best_class['model']}\n")
        f.write(f"Best Accuracy: {best_class['accuracy']:.2f}%\n")
        f.write(f"Mean Accuracy: {df_class['accuracy'].mean():.2f}%\n")
        f.write(f"Standard Deviation: {df_class['accuracy'].std():.2f}%\n")
        f.write(f"Accuracy Range: {df_class['accuracy'].min():.2f}% - {df_class['accuracy'].max():.2f}%\n\n")
        f.write("All Classification Models:\n")
        for idx, row in df_class.iterrows():
            prec_str = f"{row['precision']:.4f}" if pd.notna(row['precision']) else "N/A"
            rec_str = f"{row['recall']:.4f}" if pd.notna(row['recall']) else "N/A"
            f1_str = f"{row['f1_score']:.4f}" if pd.notna(row['f1_score']) else "N/A"
            f.write(f"  {idx+1}. {row['model']}: Accuracy={row['accuracy']:.2f}%, "
                   f"Precision={prec_str}, Recall={rec_str}, "
                   f"F1={f1_str}\n")
        f.write("\n")
        
        # Cavity Detection Section
        f.write("2. CAVITY DETECTION MODELS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {len(df_cavity)}\n")
        f.write(f"Best Model: {best_cavity['model']}\n")
        f.write(f"Best mAP50: {best_cavity['mAP50']:.4f}\n")
        f.write(f"Mean mAP50: {df_cavity['mAP50'].mean():.4f}\n")
        f.write(f"Standard Deviation: {df_cavity['mAP50'].std():.4f}\n")
        f.write(f"Mean Precision: {df_cavity['precision'].mean():.4f}\n")
        f.write(f"Mean Recall: {df_cavity['recall'].mean():.4f}\n\n")
        f.write("All Cavity Detection Models:\n")
        for idx, row in df_cavity.iterrows():
            f.write(f"  {idx+1}. {row['model']}: mAP50={row['mAP50']:.4f}, "
                   f"mAP50-95={row['mAP50-95']:.4f}, Precision={row['precision']:.4f}, "
                   f"Recall={row['recall']:.4f}\n")
        f.write("\n")
        
        # OPG Detection Section
        f.write("3. OPG DETECTION MODELS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {len(df_opg)}\n")
        f.write(f"Best Model: {best_opg['model']}\n")
        f.write(f"Best mAP50: {best_opg['mAP50']:.4f}\n")
        f.write(f"Mean mAP50: {df_opg['mAP50'].mean():.4f}\n")
        f.write(f"Standard Deviation: {df_opg['mAP50'].std():.4f}\n")
        f.write(f"Mean Precision: {df_opg['precision'].mean():.4f}\n")
        f.write(f"Mean Recall: {df_opg['recall'].mean():.4f}\n\n")
        f.write("All OPG Detection Models:\n")
        for idx, row in df_opg.iterrows():
            f.write(f"  {idx+1}. {row['model']}: mAP50={row['mAP50']:.4f}, "
                   f"mAP50-95={row['mAP50-95']:.4f}, Precision={row['precision']:.4f}, "
                   f"Recall={row['recall']:.4f}\n")
        f.write("\n")
        
        # Segmentation Section
        f.write("4. SEGMENTATION MODELS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {len(df_seg)}\n")
        f.write(f"Best Model: {best_seg['model']}\n")
        f.write(f"Best IoU: {best_seg['mean_iou']:.4f}\n")
        f.write(f"Mean IoU: {df_seg['mean_iou'].mean():.4f}\n")
        f.write(f"Standard Deviation: {df_seg['mean_iou'].std():.4f}\n")
        f.write(f"IoU Range: {df_seg['mean_iou'].min():.4f} - {df_seg['mean_iou'].max():.4f}\n\n")
        f.write("All Segmentation Models:\n")
        for idx, row in df_seg.iterrows():
            f.write(f"  {idx+1}. {row['model']}: Mean IoU={row['mean_iou']:.4f}\n")
        f.write("\n")
        
        # Clinical Classification Section
        f.write("5. CLINICAL CLASSIFICATION MODELS ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Models Evaluated: {len(df_clinical)}\n")
        f.write(f"Best Model: {best_clinical['model']}\n")
        if pd.notna(best_clinical['accuracy']):
            f.write(f"Best Accuracy: {best_clinical['accuracy']:.2f}%\n")
            f.write(f"Mean Accuracy: {df_clinical['accuracy'].mean():.2f}%\n")
        if pd.notna(best_clinical['f1_score']):
            f.write(f"Best F1-Score: {best_clinical['f1_score']:.4f}\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR PRODUCTION\n")
        f.write("-"*80 + "\n")
        f.write(f"1. Classification Task: Use {best_class['model']}\n")
        f.write(f"   - Accuracy: {best_class['accuracy']:.2f}%\n")
        f.write(f"   - F1-Score: {best_class['f1_score']:.4f}\n\n")
        f.write(f"2. Cavity Detection: Use {best_cavity['model']}\n")
        f.write(f"   - mAP50: {best_cavity['mAP50']:.4f}\n")
        f.write(f"   - Precision: {best_cavity['precision']:.4f}\n")
        f.write(f"   - Recall: {best_cavity['recall']:.4f}\n\n")
        f.write(f"3. OPG Detection: Use {best_opg['model']}\n")
        f.write(f"   - mAP50: {best_opg['mAP50']:.4f}\n")
        f.write(f"   - Precision: {best_opg['precision']:.4f}\n")
        f.write(f"   - Recall: {best_opg['recall']:.4f}\n\n")
        f.write(f"4. Segmentation: Use {best_seg['model']}\n")
        f.write(f"   - Mean IoU: {best_seg['mean_iou']:.4f}\n\n")
        f.write(f"5. Clinical Photo Classification: Use {best_clinical['model']}\n")
        if pd.notna(best_clinical['accuracy']):
            f.write(f"   - Accuracy: {best_clinical['accuracy']:.2f}%\n")
        if pd.notna(best_clinical['f1_score']):
            f.write(f"   - F1-Score: {best_clinical['f1_score']:.4f}\n")
        
        f.write("\n")
        f.write("OUTPUT FILES\n")
        f.write("-"*80 + "\n")
        f.write(f"All analysis results have been saved to: {RESEARCH_OUTPUT_DIR}\n")
        f.write(f"  - Tables: {os.path.join(RESEARCH_OUTPUT_DIR, 'tables')}\n")
        f.write(f"  - Visualizations: {os.path.join(RESEARCH_OUTPUT_DIR, 'visualizations')}\n")
        f.write(f"  - Data: {os.path.join(RESEARCH_OUTPUT_DIR, 'data')}\n")
        f.write("\n" + "="*80 + "\n")
    
    return report_path

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL ANALYSIS FOR RESEARCH PAPER")
    print("="*70)
    print(f"Output Directory: {RESEARCH_OUTPUT_DIR}\n")
    
    # Run all analyses
    best_class, df_class = analyze_classification()
    best_cavity, best_opg, df_cavity, df_opg = analyze_detection()
    best_seg, df_seg = analyze_segmentation()
    best_clinical, df_clinical = analyze_clinical()
    
    # Generate comprehensive report
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE RESEARCH REPORT")
    print("="*70)
    report_path = generate_research_report(best_class, df_class, 
                                          best_cavity, best_opg, df_cavity, df_opg,
                                          best_seg, df_seg, 
                                          best_clinical, df_clinical)
    print(f"✓ Report generated: {report_path}")
    
    # Create summary table
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Format metric displays
    class_metric = f"Accuracy: {best_class['accuracy']:.2f}%"
    cavity_metric = f"mAP50: {best_cavity['mAP50']:.4f}"
    opg_metric = f"mAP50: {best_opg['mAP50']:.4f}"
    seg_metric = f"IoU: {best_seg['mean_iou']:.4f}"
    clin_metric = f"Accuracy: {best_clinical['accuracy']:.2f}%" if pd.notna(best_clinical['accuracy']) else "N/A"
    
    summary = pd.DataFrame([
        {'Task': 'Classification', 'Best Model': best_class['model'], 
         'Primary Metric': class_metric},
        {'Task': 'Cavity Detection', 'Best Model': best_cavity['model'], 
         'Primary Metric': cavity_metric},
        {'Task': 'OPG Detection', 'Best Model': best_opg['model'], 
         'Primary Metric': opg_metric},
        {'Task': 'Segmentation', 'Best Model': best_seg['model'], 
         'Primary Metric': seg_metric},
        {'Task': 'Clinical Classification', 'Best Model': best_clinical['model'], 
         'Primary Metric': clin_metric}
    ])
    
    print(summary.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(RESEARCH_OUTPUT_DIR, 'reports', 'final_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n✓ Summary saved: {summary_path}")
    
    # Create a comprehensive JSON report
    json_report = {
        'analysis_date': datetime.now().isoformat(),
        'output_directory': RESEARCH_OUTPUT_DIR,
        'models': {
            'classification': {
                'best_model': best_class['model'],
                'accuracy': float(best_class['accuracy']) if pd.notna(best_class['accuracy']) else None,
                'all_models': df_class.where(pd.notna(df_class), None).to_dict('records')
            },
            'cavity_detection': {
                'best_model': best_cavity['model'],
                'mAP50': float(best_cavity['mAP50']) if pd.notna(best_cavity['mAP50']) else None,
                'all_models': df_cavity.where(pd.notna(df_cavity), None).to_dict('records')
            },
            'opg_detection': {
                'best_model': best_opg['model'],
                'mAP50': float(best_opg['mAP50']) if pd.notna(best_opg['mAP50']) else None,
                'all_models': df_opg.where(pd.notna(df_opg), None).to_dict('records')
            },
            'segmentation': {
                'best_model': best_seg['model'],
                'mean_iou': float(best_seg['mean_iou']) if pd.notna(best_seg['mean_iou']) else None,
                'all_models': df_seg.where(pd.notna(df_seg), None).to_dict('records')
            },
            'clinical_classification': {
                'best_model': best_clinical['model'],
                'accuracy': float(best_clinical['accuracy']) if pd.notna(best_clinical['accuracy']) else None,
                'all_models': df_clinical.where(pd.notna(df_clinical), None).to_dict('records')
            }
        }
    }
    
    json_path = os.path.join(RESEARCH_OUTPUT_DIR, 'reports', 'analysis_report.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"✓ JSON report saved: {json_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {RESEARCH_OUTPUT_DIR}")
    print(f"  - Tables (CSV, HTML, LaTeX): tables/")
    print(f"  - Visualizations (PNG): visualizations/")
    print(f"  - Raw Data (CSV): data/")
    print(f"  - Reports (TXT, JSON): reports/")

if __name__ == '__main__':
    main()
