import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = r'model4_clinical_classification\best_clinical_photo_model.pth'
DATA_DIR = r'datasets\normal photographs\Dental diseases_Model'
OUTPUT_DIR = r'outputs\model4_clinical_outputs'
NUM_SAMPLES = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
CLASS_NAMES = ['Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth_discoloration_augmented']

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Collect sample images
print("Collecting sample images...")
image_paths = []
for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_dir):
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_paths.extend(images[:NUM_SAMPLES // len(CLASS_NAMES)])

# Generate predictions
print(f"Generating predictions for {len(image_paths)} images...")
for idx, img_path in enumerate(image_paths):
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
        
        # Get true label
        true_label = os.path.basename(os.path.dirname(img_path))
        pred_label = CLASS_NAMES[pred_idx]
        
        # Create visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax[0].imshow(img)
        ax[0].axis('off')
        ax[0].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}', 
                       fontsize=10, color='green' if true_label == pred_label else 'red')
        
        # Display probabilities
        ax[1].barh(CLASS_NAMES, probs.cpu().numpy())
        ax[1].set_xlabel('Probability')
        ax[1].set_title('Class Probabilities')
        ax[1].set_xlim([0, 1])
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'prediction_{idx+1:03d}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[{idx+1}/{len(image_paths)}] Saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"\nCompleted! Outputs saved to: {OUTPUT_DIR}")
