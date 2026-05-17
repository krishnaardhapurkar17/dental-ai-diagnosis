import json
import numpy as np
from pathlib import Path

# Check a few annotation files
ann_dir = Path("datasets/teeth-segmentation-on-dental-x-ray-images-DatasetNinja/ds/ann")
meta_path = Path("datasets/teeth-segmentation-on-dental-x-ray-images-DatasetNinja/meta.json")

with open(meta_path) as f:
    meta = json.load(f)

print(f"Total classes in meta: {len(meta['classes'])}")
print(f"Class IDs: {[c['id'] for c in meta['classes'][:5]]}")

# Check first 3 annotations
for i, ann_file in enumerate(sorted(ann_dir.glob("*.json"))[:3]):
    with open(ann_file) as f:
        ann = json.load(f)
    
    print(f"\n{ann_file.name}:")
    print(f"  Objects: {len(ann['objects'])}")
    print(f"  Image size: {ann['size']['width']}x{ann['size']['height']}")
    
    if ann['objects']:
        obj = ann['objects'][0]
        print(f"  First object classId: {obj['classId']}")
        print(f"  Geometry type: {obj['geometryType']}")
        if obj['geometryType'] == 'bitmap':
            print(f"  Bitmap origin: {obj['bitmap']['origin']}")
