#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRNano


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on football dataset")
    parser.add_argument(
        "--model",
        type=str,
        choices=["nano", "small", "medium"],
        default="medium",
        help="Model size (nano/small/medium)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="football_dataset_coco",
        help="Path to COCO format dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/rfdetr/football",
        help="Output directory for training results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1280,
        help="Input image resolution (higher is better for small objects)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_classes = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
    }

    print("=" * 80)
    print("RF-DETR Football Detection Training")
    print("=" * 80)
    print(f"Model:          RF-DETR-{args.model.capitalize()}")
    print(f"Dataset:        {args.dataset_dir}")
    print(f"Output:         {args.output_dir}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Resolution:     {args.resolution}")
    print(f"Learning rate:  {args.lr}")
    print(f"Device:         {args.device}")
    print(f"Grad accum:     {args.grad_accum_steps}")
    print(f"Early stopping: {args.early_stopping_patience} epochs")
    print("=" * 80)
    print()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_dir}\n"
            "Please run convert_yolo_to_coco.py first to create the COCO dataset."
        )

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_ann = train_dir / "_annotations.coco.json"
    val_ann = val_dir / "_annotations.coco.json"

    if not train_ann.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_ann}")

    if not val_ann.exists():
        raise FileNotFoundError(f"Validation annotations not found: {val_ann}")

    print("✓ Dataset structure verified")
    print(f"  Train: {train_ann}")
    print(f"  Val:   {val_ann}")
    print()

    print("Initializing model...")
    ModelClass = model_classes[args.model]
    model = ModelClass()
    print(f"✓ Model initialized: {model.__class__.__name__}")
    print()

    training_history = []

    def log_epoch(data):
        """Callback to log epoch metrics."""
        training_history.append(data)
        print(f"  Epoch {data.get('epoch', '?')}: {data}")

    model.callbacks["on_fit_epoch_end"].append(log_epoch)

    print("Starting training...")
    print("=" * 80)

    try:
        model.train(
            dataset_dir=str(args.dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            lr_encoder=args.lr / 10, 
            output_dir=str(args.output_dir),
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            tensorboard=True,
            early_stopping=True,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            use_ema=True,  
            amp=True,  
            weight_decay=1e-4,
            num_workers=args.num_workers,
        )

        print()
        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"✓ Model saved to: {args.output_dir}")
        print(f"✓ Best weights: {args.output_dir}/weights/best.pt")
        print()

        if training_history:
            print("Training Summary:")
            print(f"  Total epochs: {len(training_history)}")
            print(f"  Final metrics: {training_history[-1]}")
            print()

    except Exception as e:
        print()
        print("=" * 80)
        print("TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        raise

    print("Next steps:")
    print("  1. Evaluate on test set:")
    print(f"     python scripts/emil/evaluate_rfdetr.py --model {args.output_dir}/weights/best.pt")
    print("  2. Compare with YOLO results")
    print("=" * 80)


if __name__ == "__main__":
    main()
