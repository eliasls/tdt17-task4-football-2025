from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import json

print("=" * 80)
print("YOLO11n TRAINING - COMPLETE FOOTBALL DATASET")
print("=" * 80)

# Check GPU
print(f"\nEnvironment:")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Paths
WORKSPACE = Path("/cluster/work/emilkl/tdt17-task4-football-2025")
DATA_YAML = WORKSPACE / "football_dataset_complete" / "data.yaml"
SPLIT_INFO = WORKSPACE / "football_dataset_complete" / "split_info.json"

# Verify dataset
print(f"\n{'='*80}")
print("DATASET INFO")
print(f"{'='*80}")

assert DATA_YAML.exists(), f"data.yaml not found! Run setup_complete_dataset.py first"

with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)

with open(SPLIT_INFO, 'r') as f:
    split_info = json.load(f)

stats = split_info['statistics']
print(f"Total images: {stats['total_images']:,}")
print(f"  Train: {stats['train_images']:,} ({stats['train_images']/stats['total_images']*100:.1f}%)")
print(f"  Val:   {stats['val_images']:,} ({stats['val_images']/stats['total_images']*100:.1f}%)")
if stats['test_images'] > 0:
    print(f"  Test:  {stats['test_images']:,} ({stats['test_images']/stats['total_images']*100:.1f}%)")

print(f"\nClasses: {list(data_config['names'].values())}")
print(f"\nMatches in training: {split_info['split_config']['train']}")
print(f"Matches in validation: {split_info['split_config']['val']}")

# Training configuration
print(f"\n{'='*80}")
print("TRAINING CONFIGURATION")
print(f"{'='*80}")

config = {
    'model': 'yolo11n.pt',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,  # Adjust if GPU OOM
    'device': 0,
    'workers': 8,
    'project': 'football_yolo11',
    'name': 'complete_dataset_v1',
    'patience': 20,
    'save_period': 10,
}

for key, value in config.items():
    print(f"  {key:15} = {value}")

# Load model
print(f"\n{'='*80}")
print("INITIALIZING MODEL")
print(f"{'='*80}")
model = YOLO(config['model'])
print(f"✓ Loaded {config['model']}")
print(f"  Parameters: 2.6M")
print(f"  FLOPs: 6.5G")

# Start training
print(f"\n{'='*80}")
print("STARTING TRAINING")
print(f"{'='*80}")
print(f"Expected time on A100: ~5-6 hours")
print(f"{'='*80}\n")

results = model.train(
    data=str(DATA_YAML),
    epochs=config['epochs'],
    imgsz=config['imgsz'],
    batch=config['batch'],
    device=config['device'],
    workers=config['workers'],
    project=config['project'],
    name=config['name'],
    
    # Early stopping & checkpointing
    patience=config['patience'],
    save=True,
    save_period=config['save_period'],
    
    # Validation & visualization
    val=True,
    plots=True,
    
    # Data augmentation (optimized for football)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    mosaic=1.0,
    mixup=0.0,
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    # Other
    cache=False,
    verbose=True,
)

# Final validation
print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}")

best_model_path = f"{config['project']}/{config['name']}/weights/best.pt"
print(f"\nBest model: {best_model_path}")

print("\nRunning final validation...")
best_model = YOLO(best_model_path)
metrics = best_model.val(data=str(DATA_YAML))

print(f"\n{'='*80}")
print("FINAL METRICS")
print(f"{'='*80}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

# Per-class metrics
print(f"\nPer-class mAP50-95:")
class_names = list(data_config['names'].values())
for i, name in enumerate(class_names):
    if i < len(metrics.box.maps):
        print(f"  {name:15} {metrics.box.maps[i]:.4f}")

print(f"\n{'='*80}")
print(f"Results saved to: {config['project']}/{config['name']}/")
print(f"{'='*80}\n")