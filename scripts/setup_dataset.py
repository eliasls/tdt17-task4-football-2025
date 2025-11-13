from pathlib import Path
import yaml
import json
from collections import Counter

# Configuration
DATASET_ROOT = Path("/cluster/projects/vc/courses/TDT17/other/Football2025")
WORKSPACE = Path("/cluster/work/emilkl/tdt17-task4-football-2025")
FOOTBALL_DATASET = WORKSPACE / "football_dataset_complete"

# SPLIT CONFIGURATION - Edit here to change splits
SPLIT_CONFIG = {
    'train': ['RBK-AALESUND', 'RBK-FREDRIKSTAD', 'RBK-BODO'],
    'val': ['RBK-HamKam'],
    'test': ['RBK-VIKING']
}

print("=" * 80)
print("COMPLETE FOOTBALL DATASET SETUP")
print("=" * 80)
print(f"Total matches: 5 (RBK-BODO has 3 parts)")
print(f"Output: {FOOTBALL_DATASET}")

# Step 1: Discover all data
print("\n[1/5] Discovering all matches and parts...")

match_parts = {}

# Standard structure matches
for match_name in ['RBK-AALESUND', 'RBK-FREDRIKSTAD', 'RBK-HamKam', 'RBK-VIKING']:
    match_dir = DATASET_ROOT / match_name
    if not match_dir.exists():
        print(f"  ⚠️  Skipping {match_name} (not found)")
        continue
    
    img_dir = match_dir / "data/images/train"
    lbl_dir = match_dir / "labels/train"
    
    if img_dir.exists() and lbl_dir.exists():
        match_parts[match_name] = [{
            'name': match_name,
            'img_dir': img_dir,
            'lbl_dir': lbl_dir
        }]
        print(f"  ✓ Found {match_name}")

# RBK-BODO with nested structure
bodo_dir = DATASET_ROOT / "RBK-BODO"
if bodo_dir.exists():
    bodo_parts = []
    
    for part_num in [1, 2, 3]:
        part_dir = bodo_dir / f"part{part_num}"
        
        # Find nested directory (RBK_BODO_PART1, etc.)
        nested_dirs = list(part_dir.glob("RBK_BODO_PART*"))
        
        if nested_dirs:
            data_root = nested_dirs[0]
            img_dir = data_root / "data/images/train"
            lbl_dir = data_root / "labels/train"
            
            if img_dir.exists() and lbl_dir.exists():
                bodo_parts.append({
                    'name': f'RBK-BODO-part{part_num}',
                    'img_dir': img_dir,
                    'lbl_dir': lbl_dir
                })
    
    if bodo_parts:
        match_parts['RBK-BODO'] = bodo_parts
        print(f"  ✓ Found RBK-BODO ({len(bodo_parts)} parts)")

# Step 2: Analyze each part
print("\n[2/5] Analyzing data...")

match_info = {}
total_images = 0
total_objects = 0

for match_name, parts in match_parts.items():
    print(f"\n  {match_name}:")
    
    match_total_imgs = 0
    match_total_objs = 0
    part_details = []
    
    for part in parts:
        images = list(part['img_dir'].glob("*.png"))
        labels = list(part['lbl_dir'].glob("*.txt"))
        
        # Count objects per class
        class_counts = Counter()
        for lbl in labels:
            try:
                with open(lbl, 'r') as f:
                    for line in f:
                        parts_data = line.strip().split()
                        if len(parts_data) >= 5:
                            class_id = int(float(parts_data[0]))
                            class_counts[class_id] += 1
            except Exception as e:
                print(f"    ⚠️  Error reading {lbl}: {e}")
        
        num_images = len(images)
        num_objects = sum(class_counts.values())
        
        match_total_imgs += num_images
        match_total_objs += num_objects
        
        part_details.append({
            'name': part['name'],
            'img_dir': part['img_dir'],
            'lbl_dir': part['lbl_dir'],
            'num_images': num_images,
            'num_labels': len(labels),
            'class_counts': dict(class_counts)
        })
        
        if len(parts) > 1:  # Multiple parts
            print(f"    {part['name']}: {num_images} images, {num_objects} objects")
        else:
            print(f"    {num_images} images, {num_objects} objects")
    
    match_info[match_name] = {
        'parts': part_details,
        'total_images': match_total_imgs,
        'total_objects': match_total_objs
    }
    
    total_images += match_total_imgs
    total_objects += match_total_objs

print(f"\n  {'='*70}")
print(f"  DATASET TOTALS:")
print(f"  {'='*70}")
print(f"  Total images:  {total_images:,}")
print(f"  Total objects: {total_objects:,}")
print(f"  Avg objects/image: {total_objects/total_images:.1f}")

# Step 3: Display split strategy
print("\n[3/5] Split configuration:")

split_stats = {}
for split_name, match_list in SPLIT_CONFIG.items():
    split_imgs = sum(match_info[m]['total_images'] for m in match_list if m in match_info)
    split_objs = sum(match_info[m]['total_objects'] for m in match_list if m in match_info)
    split_stats[split_name] = {'images': split_imgs, 'objects': split_objs}
    
    print(f"\n  {split_name.upper()}: {len(match_list)} matches, {split_imgs:,} images ({split_imgs/total_images*100:.1f}%)")
    for m in match_list:
        if m in match_info:
            info = match_info[m]
            print(f"    • {m}: {info['total_images']:,} images", end='')
            if len(info['parts']) > 1:
                print(f" ({len(info['parts'])} parts)")
            else:
                print()

# Step 4: Create dataset structure with symlinks
print("\n[4/5] Creating dataset structure...")
FOOTBALL_DATASET.mkdir(parents=True, exist_ok=True)

for split_name, match_list in SPLIT_CONFIG.items():
    if not match_list:
        continue
    
    print(f"\n  Creating {split_name} split...")
    
    img_dir = FOOTBALL_DATASET / "images" / split_name
    lbl_dir = FOOTBALL_DATASET / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old symlinks
    for old in list(img_dir.glob("*")):
        old.unlink()
    for old in list(lbl_dir.glob("*")):
        old.unlink()
    
    # Create symlinks for each match
    total_linked = 0
    for match_name in match_list:
        if match_name not in match_info:
            continue
        
        for part in match_info[match_name]['parts']:
            images = list(part['img_dir'].glob("*.png"))
            
            for img in images:
                # Create unique filename
                new_name = f"{part['name']}_{img.name}"
                
                # Image symlink
                img_link = img_dir / new_name
                img_link.symlink_to(img)
                
                # Label symlink
                lbl = part['lbl_dir'] / f"{img.stem}.txt"
                if lbl.exists():
                    lbl_link = lbl_dir / f"{part['name']}_{lbl.name}"
                    lbl_link.symlink_to(lbl)
                
                total_linked += 1
    
    print(f"    ✓ Created {total_linked:,} symlinks")

# Step 5: Create configuration files
print("\n[5/5] Creating configuration files...")

# data.yaml for YOLO
data_yaml = FOOTBALL_DATASET / "data.yaml"
data_config = {
    'path': str(FOOTBALL_DATASET),
    'train': 'images/train',
    'val': 'images/val',
    'names': {0: 'player', 1: 'ball', 2: 'event_labels'},
    'nc': 3
}

if SPLIT_CONFIG.get('test'):
    data_config['test'] = 'images/test'

with open(data_yaml, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

print(f"  ✓ Created {data_yaml}")

# Split info JSON
split_info = {
    'split_config': SPLIT_CONFIG,
    'match_details': {
        name: {
            'total_images': info['total_images'],
            'total_objects': info['total_objects'],
            'num_parts': len(info['parts']),
            'parts': [p['name'] for p in info['parts']]
        }
        for name, info in match_info.items()
    },
    'statistics': {
        'total_images': total_images,
        'total_objects': total_objects,
        'train_images': split_stats['train']['images'],
        'val_images': split_stats['val']['images'],
        'test_images': split_stats.get('test', {}).get('images', 0),
    }
}

with open(FOOTBALL_DATASET / "split_info.json", 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"  ✓ Saved split_info.json")

# Verification commands
print("\n" + "=" * 80)
print("SETUP COMPLETE! ✓")
print("=" * 80)
print(f"\nDataset location: {FOOTBALL_DATASET}")
print(f"Total images: {total_images:,}")
print(f"\nSplit breakdown:")
print(f"  Train: {split_stats['train']['images']:,} images ({split_stats['train']['images']/total_images*100:.1f}%)")
print(f"  Val:   {split_stats['val']['images']:,} images ({split_stats['val']['images']/total_images*100:.1f}%)")
if split_stats.get('test'):
    print(f"  Test:  {split_stats['test']['images']:,} images ({split_stats['test']['images']/total_images*100:.1f}%)")

print(f"\nVerify setup:")
print(f"  ls {FOOTBALL_DATASET}/images/train/ | head")
print(f"  cat {FOOTBALL_DATASET}/data.yaml")
print(f"\nNext step:")
print(f"  python train_yolo11.py")
print("=" * 80)