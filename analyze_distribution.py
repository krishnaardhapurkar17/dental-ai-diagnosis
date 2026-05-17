import yaml
from pathlib import Path
from collections import defaultdict

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def analyze_split(label_dir):
    class_counts = defaultdict(int)
    total_images = 0
    
    label_files = list(Path(label_dir).glob("*.txt"))
    total_images = len(label_files)
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1
    
    return total_images, dict(class_counts)

def main():
    config = load_config()
    base_dir = Path(config['project']['base_dir'])
    cavity_dataset = Path("d:/college/EDAI/project/datasets/Cavity Dataset")
    
    print("="*60)
    print("CAVITY DATASET DISTRIBUTION ANALYSIS")
    print("="*60)
    
    splits = ['train', 'valid', 'test']
    all_stats = {}
    
    for split in splits:
        label_dir = cavity_dataset / split / "labels"
        if not label_dir.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue
            
        n_images, class_counts = analyze_split(label_dir)
        all_stats[split] = {'images': n_images, 'classes': class_counts}
        
        total_instances = sum(class_counts.values())
        healthy = class_counts.get(0, 0)
        unhealthy = class_counts.get(1, 0)
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {n_images}")
        print(f"  Total instances: {total_instances}")
        print(f"  Healthy teeth (class 0): {healthy} ({healthy/total_instances*100:.1f}%)")
        print(f"  Unhealthy teeth (class 1): {unhealthy} ({unhealthy/total_instances*100:.1f}%)")
        print(f"  Ratio (healthy:unhealthy): {healthy/unhealthy:.2f}:1" if unhealthy > 0 else "  Ratio: N/A")
    
    print("\n" + "="*60)
    print("DISTRIBUTION COMPARISON")
    print("="*60)
    
    for split in splits:
        if split in all_stats:
            stats = all_stats[split]
            healthy = stats['classes'].get(0, 0)
            unhealthy = stats['classes'].get(1, 0)
            total = healthy + unhealthy
            print(f"{split.upper()}: {healthy/total*100:.1f}% healthy, {unhealthy/total*100:.1f}% unhealthy")
    
    if 'test' in all_stats and 'train' in all_stats:
        test_ratio = all_stats['test']['classes'].get(0, 0) / all_stats['test']['classes'].get(1, 1)
        train_ratio = all_stats['train']['classes'].get(0, 0) / all_stats['train']['classes'].get(1, 1)
        print(f"\nDistribution mismatch: Train ratio {train_ratio:.2f}:1 vs Test ratio {test_ratio:.2f}:1")

if __name__ == "__main__":
    main()
