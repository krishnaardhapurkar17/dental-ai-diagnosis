import os
import json
import base64
import zlib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ClassificationDataset(Dataset):
    """Dataset for classification task (folder structure: root/class_name/image.jpg)"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Load all image paths and labels
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                try:
                    # Validate image can be opened
                    Image.open(img_path).convert('RGB')
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                except Exception as e:
                    print(f"⚠️  Skipping corrupted image: {img_path}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (300, 300))), label
            return Image.new('RGB', (300, 300)), label


class SegmentationDataset(Dataset):
    """Dataset for segmentation task (Supervisely JSON format)"""
    
    def __init__(self, img_dir, ann_dir, meta_json, transform=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        
        # Load class mapping from meta.json
        with open(meta_json, 'r') as f:
            meta = json.load(f)
        
        # Map classId to tooth number (1-32)
        self.class_id_to_label = {}
        for cls in meta['classes']:
            tooth_num = int(cls['title'])
            self.class_id_to_label[cls['id']] = tooth_num
        
        # Get all image files
        self.image_files = sorted([f for f in self.img_dir.glob("*.jpg")])
        
        # Validate that annotations exist
        valid_images = []
        for img_file in self.image_files:
            ann_file = self.ann_dir / f"{img_file.name}.json"
            if ann_file.exists():
                valid_images.append(img_file)
            else:
                print(f"⚠️  Missing annotation for {img_file.name}")
        
        self.image_files = valid_images
    
    def __len__(self):
        return len(self.image_files)
    
    def _decode_bitmap(self, bitmap_data, origin, img_height, img_width):
        """Decode Supervisely bitmap mask"""
        try:
            # Decode base64 and decompress
            decoded = base64.b64decode(bitmap_data)
            decoded = zlib.decompress(decoded)
            
            # Parse PNG bitmap
            from io import BytesIO
            bitmap_img = Image.open(BytesIO(decoded))
            bitmap_array = np.array(bitmap_img)
            
            # Create full-size mask
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Place bitmap at origin position
            h, w = bitmap_array.shape[:2] if len(bitmap_array.shape) > 1 else (0, 0)
            if h == 0 or w == 0:
                return mask
            
            y, x = origin
            
            # Ensure valid boundaries
            if y < 0 or x < 0 or y >= img_height or x >= img_width:
                return mask
            
            y_end = min(y + h, img_height)
            x_end = min(x + w, img_width)
            h_crop = y_end - y
            w_crop = x_end - x
            
            if h_crop > 0 and w_crop > 0:
                mask[y:y_end, x:x_end] = bitmap_array[:h_crop, :w_crop] > 0
            
            return mask
        except Exception as e:
            return np.zeros((img_height, img_width), dtype=np.uint8)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        ann_file = self.ann_dir / f"{img_file.name}.json"
        
        try:
            # Load image
            image = Image.open(img_file).convert('RGB')
            img_width, img_height = image.size
            
            # Load annotation
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            
            # Create empty mask (background = 0)
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Parse each tooth object - BINARY: all teeth = 1, background = 0
            for obj in ann['objects']:
                if obj['geometryType'] == 'bitmap':
                    bitmap_data = obj['bitmap']['data']
                    origin = obj['bitmap']['origin']
                    
                    # Decode bitmap and add to mask (all teeth get label 1)
                    tooth_mask = self._decode_bitmap(bitmap_data, origin, img_height, img_width)
                    mask[tooth_mask > 0] = 1  # Binary: tooth = 1
            
            # Convert to numpy arrays
            image = np.array(image)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask'].long()  # Ensure LongTensor
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).long()
            
            return image, mask
            
        except Exception as e:
            # Return black image and empty mask as fallback
            if self.transform:
                dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
                dummy_mask = np.zeros((512, 512), dtype=np.uint8)
                transformed = self.transform(image=dummy_img, mask=dummy_mask)
                return transformed['image'], transformed['mask'].long()
            return torch.zeros(3, 512, 512), torch.zeros(512, 512, dtype=torch.long)
