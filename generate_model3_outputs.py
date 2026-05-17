import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = r'model3_segmentation\checkpoints\best_model.pth'
DATA_DIR = r'model3_segmentation\data\test\img'
OUTPUT_DIR = r'outputs\model3_segmentation_outputs'
NUM_SAMPLES = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
model = smp.Unet(
    encoder_name='efficientnet-b3',
    encoder_weights=None,
    in_channels=3,
    classes=2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

print("Collecting sample images...")
image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:NUM_SAMPLES]

print(f"Generating segmentation outputs for {len(image_files)} images...")
for idx, img_file in enumerate(image_files):
    try:
        img_path = os.path.join(DATA_DIR, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=img_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
        
        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                       (img_rgb.shape[1], img_rgb.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_mask_resized, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = img_rgb.copy()
        overlay[pred_mask_resized > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        axes[2].imshow(blended)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f'segmentation_{idx+1:03d}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"[{idx+1}/{len(image_files)}] Saved: {output_path}")
        
    except Exception as e:
        print(f"Error processing {img_file}: {e}")

print(f"\nCompleted! Outputs saved to: {OUTPUT_DIR}")
