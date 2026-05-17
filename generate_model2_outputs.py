import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

CAVITY_MODEL = r'dental_diagnosis_dashboard\models\best_cavity_detection.pt'
OPG_MODEL = r'dental_diagnosis_dashboard\models\best_opg_detection.pt'
DATA_DIR = r'datasets\normal photographs\Dental diseases_Model'
OUTPUT_DIR = r'outputs\model2_detection_outputs'
NUM_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'cavity'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'opg'), exist_ok=True)

# Collect images from all classes
print("Collecting clinical photo images...")
all_images = []
for class_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(class_dir):
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        all_images.extend(images[:NUM_SAMPLES // 5])

# Cavity Detection
print(f"\nLoading cavity detection model...")
cavity_model = YOLO(CAVITY_MODEL)

print(f"Generating cavity detection outputs on {len(all_images)} clinical photos...")
for idx, img_path in enumerate(all_images):
    results = cavity_model.predict(img_path, conf=0.25, save=False)
    
    img_original = cv2.imread(img_path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_detected = img_original.copy()
    
    detection_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"
            
            color = (0, 255, 0) if result.names[cls] == 'healthy_teeth' else (255, 0, 0)
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_detected, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            detection_count += 1
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_detected)
    axes[1].set_title(f'Cavity Detection ({detection_count} detections)', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'cavity', f'cavity_{idx+1:03d}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"[{idx+1}/{len(all_images)}] Saved: {output_path}")

# OPG Detection
print(f"\nLoading OPG detection model...")
opg_model = YOLO(OPG_MODEL)

print(f"Generating OPG detection outputs on {len(all_images)} clinical photos...")
for idx, img_path in enumerate(all_images):
    results = opg_model.predict(img_path, conf=0.25, save=False)
    
    img_original = cv2.imread(img_path)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_detected = img_original.copy()
    
    detection_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"
            
            cv2.rectangle(img_detected, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_detected, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            detection_count += 1
    
    # Create side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(img_original)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(img_detected)
    axes[1].set_title(f'OPG Detection ({detection_count} detections)', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'opg', f'opg_{idx+1:03d}.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"[{idx+1}/{len(all_images)}] Saved: {output_path}")

print(f"\nCompleted! Outputs saved to: {OUTPUT_DIR}")
