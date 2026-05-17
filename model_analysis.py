import pandas as pd
import json
from pathlib import Path

base_path = Path("D:/college/EDAI/final project/comparative_models")

categories = {
    "model1_classification": {
        "file": "comparative_results.csv",
        "metrics": ["accuracy", "f1_score"],
        "primary": "f1_score"
    },
    "model2_detection": {
        "file": "comparative_results.csv",
        "metrics": ["mAP50", "mAP50-95"],
        "primary": "mAP50"
    },
    "model3_segmentation": {
        "file": "comparative_results.csv",
        "metrics": ["mean_iou"],
        "primary": "mean_iou"
    },
    "model4_clinical": {
        "file": "comparative_results.csv",
        "metrics": ["accuracy", "f1_score"],
        "primary": "f1_score"
    }
}

results = {}

for category, config in categories.items():
    csv_path = base_path / category / config["file"]
    df = pd.read_csv(csv_path)
    
    primary_metric = config["primary"]
    best_idx = df[primary_metric].idxmax()
    best_model = df.loc[best_idx]
    
    results[category] = {
        "best_model": best_model["model"],
        "metrics": best_model[config["metrics"]].to_dict(),
        "all_models": df.to_dict("records")
    }

print("\n" + "="*80)
print("MODEL SELECTION ANALYSIS - BEST MODELS PER CATEGORY")
print("="*80 + "\n")

for category, data in results.items():
    print(f"\n{category.upper()}")
    print("-" * 80)
    print(f"BEST MODEL: {data['best_model']}")
    print(f"  Metrics: {data['metrics']}")
    print(f"\n  All Models:")
    for model in data['all_models']:
        print(f"    - {model['model']}: {model}")

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

recommendations = {
    "Classification": results["model1_classification"]["best_model"],
    "Detection": results["model2_detection"]["best_model"],
    "Segmentation": results["model3_segmentation"]["best_model"],
    "Clinical": results["model4_clinical"]["best_model"]
}

for task, model in recommendations.items():
    print(f"{task:20} -> {model}")

with open("D:/college/EDAI/final project/model_selection_report.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDetailed report saved to: model_selection_report.json")
