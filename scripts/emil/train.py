#!/usr/bin/env python3
"""
Simple and minimal YOLO training script for football dataset.
Easy to customize via command-line arguments.
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model on football dataset')

    # Model and data
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Model to train (e.g., yolo11n.pt, yolo11s.pt)')
    parser.add_argument('--data', type=str, default='football_dataset_complete/data.yaml', help='Path to data.yaml')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (must be multiple of 32)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device (0, 1, etc.) or cpu')

    # Output
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')

    # Advanced options
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Data:       {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device:     {args.device}")
    print(f"Project:    {args.project}")
    print(f"Name:       {args.name}")
    print("=" * 60)

    # GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    # Load model and train
    print("\nLoading model...")
    model = YOLO(args.model)

    print("\nStarting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    # Get save directory
    if hasattr(results, 'save_dir'):
        save_dir = Path(results.save_dir)
    elif hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir'):
        save_dir = Path(model.trainer.save_dir)
    else:
        save_dir = Path(args.project) / args.name

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: {save_dir}")

    # Run final validation on best model
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    if best_weights.exists():
        print(f"Best weights: {best_weights}")
        print("\nRunning final validation on best model...")

        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=args.data)

        print("\n" + "=" * 60)
        print("FINAL METRICS (Best Model)")
        print("=" * 60)
        print(f"mAP50-95:  {metrics.box.map:.4f}")
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP75:     {metrics.box.map75:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")

        # Per-class metrics
        if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
            print("\nPer-class mAP50-95:")
            class_names = ['player', 'ball', 'event_labels']  # Football dataset classes
            for i, name in enumerate(class_names):
                if i < len(metrics.box.maps):
                    print(f"  {name:15} {metrics.box.maps[i]:.4f}")

        # Per-class AP50
        if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
            print("\nPer-class mAP50:")
            for i, name in enumerate(class_names):
                if i < len(metrics.box.ap50):
                    print(f"  {name:15} {metrics.box.ap50[i]:.4f}")

        print("=" * 60)
        print(f"\nOutputs:")
        print(f"  Best model:   {best_weights}")
        print(f"  Last model:   {last_weights}")
        print(f"  Results:      {save_dir}")
        print(f"  Plots:        {save_dir / 'results.png'}")
        print(f"  Confusion:    {save_dir / 'confusion_matrix.png'}")
        print(f"  Predictions:  {save_dir / 'val_batch*_pred.jpg'}")

    elif last_weights.exists():
        print(f"\nBest weights not found, using last weights: {last_weights}")
    else:
        print("\nNo model weights found!")

    print("=" * 60)

if __name__ == "__main__":
    main()
