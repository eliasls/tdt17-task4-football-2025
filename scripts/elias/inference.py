#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on football dataset')

    parser.add_argument('--model', type=str,
                        default='runs/train/elias/yolo11s_1920_rect_e150/weights/best.pt',
                        help='Model to use')
    parser.add_argument('--data', type=str,
                        default='datasets/data.yaml',
                        help='Path to data.yaml')

    parser.add_argument('--project', type=str,
                        default='runs/infere/',
                        help='Project directory for val results')
    parser.add_argument('--name', type=str,
                        default='yolo11s_1920_rect_e150_test',
                        help='Run name for val results')

    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO Inference (TEST)")
    print("=" * 60)
    print(f"Model:        {args.model}")
    print(f"Data:         {args.data}")
    print(f"Project:      {args.project}")
    print(f"Name:         {args.name}")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")

    print("\nLoading model...")
    model = YOLO(args.model)
    
    print("\nRunning evaluation on TEST set...")
    metrics = model.val(
        data=args.data,
        split="test",
        verbose=True,
        plots=True,           
        save=True,            
        project=args.project,
        name=args.name,
    )

    print("\n" + "=" * 60)
    print("TEST METRICS (Best Model)")
    print("=" * 60)
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP75:     {metrics.box.map75:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    
    class_names = list(model.names.values())

    if hasattr(metrics.box, 'p') and len(metrics.box.p) > 0:
        print("\nPer-class Precision:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.p):
                print(f"  {name:15} {metrics.box.p[i]:.4f}")
                
    if hasattr(metrics.box, 'r') and len(metrics.box.r) > 0:
        print("\nPer-class Recall:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.r):
                print(f"  {name:15} {metrics.box.r[i]:.4f}")

    if hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        print("\nPer-class mAP50:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"  {name:15} {metrics.box.ap50[i]:.4f}")
    
    if hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 0:
        print("\nPer-class mAP50-95:")
        for i, name in enumerate(class_names):
            if i < len(metrics.box.maps):
                print(f"  {name:15} {metrics.box.maps[i]:.4f}")
    
    save_dir = Path(metrics.save_dir)
    print("=" * 60)
    print(f"\nWeights used: {args.model}")
    print(f"Results dir:  {save_dir}")
    print(f"PR curve:     {save_dir / 'PR_curve.png'}")
    print(f"F1 curve:     {save_dir / 'F1_curve.png'}")
    print(f"Confusion:    {save_dir / 'confusion_matrix.png'}")
    print(f"Results plot: {save_dir / 'results.png'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
