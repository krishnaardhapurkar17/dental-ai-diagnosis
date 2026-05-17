from pathlib import Path

data_dir = Path('model1_classification/data')
for split in ['train', 'val', 'test']:
    print(f"\n{split.upper()}:")
    total = 0
    for cls_dir in sorted((data_dir / split).iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob('*.jpg')))
            print(f"  {cls_dir.name}: {count}")
            total += count
    print(f"  TOTAL: {total}")
