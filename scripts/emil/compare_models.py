#!/usr/bin/env python3
"""
Compare two trained models side-by-side.
Creates metric comparisons and 2x2 grid visualizations.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SLURM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from ultralytics import YOLO
import yaml
import pandas as pd
import json

CLASS_NAMES = {0: 'player', 1: 'ball', 2: 'event_labels'}
CLASS_COLORS = {
    'gt': 'green',
    'model1': 'red',
    'model2': 'blue'
}

def parse_yolo_label(label_path):
    """Parse YOLO format label file."""
    annotations = []
    if not label_path.exists():
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls, x, y, w, h = map(float, parts[:5])
                annotations.append({
                    'class': int(cls),
                    'bbox': [x, y, w, h],
                })
    return annotations

def bbox_iou(box1, box2):
    """Calculate IoU between two bounding boxes (normalized xywh format)."""
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2

    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_image_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate metrics for a single image."""
    if len(gt_boxes) == 0:
        fp = len(pred_boxes)
        return {'tp': 0, 'fp': fp, 'fn': 0, 'precision': 0 if fp > 0 else 1, 'recall': 1, 'f1': 0}

    if len(pred_boxes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_boxes), 'precision': 1, 'recall': 0, 'f1': 0}

    matched_gt = set()
    matched_pred = set()

    for i, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for j, gt in enumerate(gt_boxes):
            if gt['class'] != pred['class']:
                continue
            if j in matched_gt:
                continue

            iou = bbox_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

def draw_boxes_on_axis(ax, img, boxes, color, title):
    """Draw bounding boxes on a matplotlib axis."""
    ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    h, w = img.shape[:2]

    for box in boxes:
        x, y, bw, bh = box['bbox']
        x1 = (x - bw/2) * w
        y1 = (y - bh/2) * h
        width = bw * w
        height = bh * h

        rect = patches.Rectangle((x1, y1), width, height,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        label = CLASS_NAMES[box['class']]
        if 'conf' in box:
            label += f" {box['conf']:.2f}"

        ax.text(x1, y1-5, label, color=color, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

def create_comparison_grid(img, gt_boxes, pred1_boxes, pred2_boxes, save_path, model1_name, model2_name):
    """Create 2x2 grid: Raw | GT | Model1 | Model2"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    # Top-left: Raw image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Raw Frame', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Top-right: Ground Truth
    draw_boxes_on_axis(axes[0, 1], img, gt_boxes, CLASS_COLORS['gt'],
                       f'Ground Truth ({len(gt_boxes)} objects)')

    # Bottom-left: Model 1
    draw_boxes_on_axis(axes[1, 0], img, pred1_boxes, CLASS_COLORS['model1'],
                       f'{model1_name} ({len(pred1_boxes)} predictions)')

    # Bottom-right: Model 2
    draw_boxes_on_axis(axes[1, 1], img, pred2_boxes, CLASS_COLORS['model2'],
                       f'{model2_name} ({len(pred2_boxes)} predictions)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def run_predictions(model, images, data_path, split):
    """Run predictions on a list of images."""
    results = []

    for img_path in images:
        # Get label path
        label_path = data_path / 'labels' / split / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path)

        # Run prediction
        pred_results = model.predict(str(img_path), verbose=False, conf=0.25)

        # Parse predictions
        pred_boxes = []
        if len(pred_results) > 0 and pred_results[0].boxes is not None:
            boxes = pred_results[0].boxes
            img_shape = pred_results[0].orig_shape
            h, w = img_shape

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                pred_boxes.append({
                    'class': cls,
                    'bbox': [x_center, y_center, width, height],
                    'conf': conf
                })

        metrics = calculate_image_metrics(gt_boxes, pred_boxes)

        results.append({
            'image_path': str(img_path),
            'image_name': img_path.name,
            'gt_boxes': gt_boxes,
            'pred_boxes': pred_boxes,
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes),
            **metrics
        })

    return results

def main():
    parser = argparse.ArgumentParser(description='Compare two trained models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model weights')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model weights')
    parser.add_argument('--name1', type=str, default='Model 1', help='Name for first model')
    parser.add_argument('--name2', type=str, default='Model 2', help='Name for second model')
    parser.add_argument('--data', type=str, default='football_dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample comparisons to visualize')
    parser.add_argument('--output-dir', type=str, default='model_comparison', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"Model 1:    {args.model1} ({args.name1})")
    print(f"Model 2:    {args.model2} ({args.name2})")
    print(f"Data:       {args.data}")
    print(f"Split:      {args.split}")
    print(f"Output:     {output_dir}")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    model1 = YOLO(args.model1)
    print(f"✓ Loaded {args.name1}")
    model2 = YOLO(args.model2)
    print(f"✓ Loaded {args.name2}")

    # Load data config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    data_path = Path(data_config['path'])
    images_dir = data_path / data_config[args.split]
    all_images = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))

    print(f"\nFound {len(all_images)} {args.split} images")

    # Run predictions with both models
    print(f"\nRunning {args.name1} predictions...")
    results1 = run_predictions(model1, all_images, data_path, args.split)
    df1 = pd.DataFrame(results1)

    print(f"Running {args.name2} predictions...")
    results2 = run_predictions(model2, all_images, data_path, args.split)
    df2 = pd.DataFrame(results2)

    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)

    metrics_comparison = {
        'Model': [args.name1, args.name2],
        'Avg Precision': [df1['precision'].mean(), df2['precision'].mean()],
        'Avg Recall': [df1['recall'].mean(), df2['recall'].mean()],
        'Avg F1': [df1['f1'].mean(), df2['f1'].mean()],
        'Total TP': [df1['tp'].sum(), df2['tp'].sum()],
        'Total FP': [df1['fp'].sum(), df2['fp'].sum()],
        'Total FN': [df1['fn'].sum(), df2['fn'].sum()],
    }

    df_metrics = pd.DataFrame(metrics_comparison)
    print(df_metrics.to_string(index=False))

    # Save metrics
    metrics_file = output_dir / 'metrics_comparison.csv'
    df_metrics.to_csv(metrics_file, index=False)
    print(f"\n✓ Metrics saved to: {metrics_file}")

    # Save detailed results
    df1.to_csv(output_dir / f'{args.name1.replace(" ", "_")}_results.csv', index=False)
    df2.to_csv(output_dir / f'{args.name2.replace(" ", "_")}_results.csv', index=False)

    # Create comparison metrics
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Find interesting cases to visualize
    # 1. Random samples
    # 2. Where model1 > model2
    # 3. Where model2 > model1
    # 4. Where both struggle

    df_combined = pd.DataFrame({
        'image_path': df1['image_path'],
        'image_name': df1['image_name'],
        'num_gt': df1['num_gt'],
        'f1_model1': df1['f1'],
        'f1_model2': df2['f1'],
        'f1_diff': df1['f1'] - df2['f1'],
    })

    # Filter to images with GT
    df_with_gt = df_combined[df_combined['num_gt'] > 0].copy()

    if len(df_with_gt) == 0:
        print("⚠ No images with ground truth found!")
        return

    # Sample different cases
    num_per_category = max(1, args.num_samples // 4)

    # Random samples
    random_samples = df_with_gt.sample(min(num_per_category, len(df_with_gt)), random_state=42)

    # Model 1 better
    model1_better = df_with_gt.nlargest(num_per_category, 'f1_diff')

    # Model 2 better
    model2_better = df_with_gt.nsmallest(num_per_category, 'f1_diff')

    # Both struggle (lowest average F1)
    df_with_gt['avg_f1'] = (df_with_gt['f1_model1'] + df_with_gt['f1_model2']) / 2
    both_struggle = df_with_gt.nsmallest(num_per_category, 'avg_f1')

    # Combine all samples
    samples_to_viz = pd.concat([random_samples, model1_better, model2_better, both_struggle]).drop_duplicates('image_name')

    print(f"\nCreating {len(samples_to_viz)} comparison visualizations...")

    for idx, row in samples_to_viz.iterrows():
        img_path = Path(row['image_path'])
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get corresponding predictions
        idx1 = df1[df1['image_path'] == row['image_path']].index[0]
        idx2 = df2[df2['image_path'] == row['image_path']].index[0]

        gt_boxes = results1[idx1]['gt_boxes']
        pred1_boxes = results1[idx1]['pred_boxes']
        pred2_boxes = results2[idx2]['pred_boxes']

        # Determine category
        if row['image_name'] in model1_better['image_name'].values:
            category = 'model1_better'
        elif row['image_name'] in model2_better['image_name'].values:
            category = 'model2_better'
        elif row['image_name'] in both_struggle['image_name'].values:
            category = 'both_struggle'
        else:
            category = 'random'

        save_path = viz_dir / f"{category}_{row['image_name']}"

        create_comparison_grid(img, gt_boxes, pred1_boxes, pred2_boxes,
                              save_path, args.name1, args.name2)

    print(f"✓ Visualizations saved to: {viz_dir}")

    # Create comparison plots
    print("\nGenerating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # F1 comparison histogram
    axes[0, 0].hist(df1['f1'], bins=30, alpha=0.5, label=args.name1, color='red')
    axes[0, 0].hist(df2['f1'], bins=30, alpha=0.5, label=args.name2, color='blue')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('F1 Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].axvline(df1['f1'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(df2['f1'].mean(), color='blue', linestyle='--', alpha=0.7)

    # Precision comparison
    axes[0, 1].hist(df1['precision'], bins=30, alpha=0.5, label=args.name1, color='red')
    axes[0, 1].hist(df2['precision'], bins=30, alpha=0.5, label=args.name2, color='blue')
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Precision Distribution')
    axes[0, 1].legend()

    # Recall comparison
    axes[1, 0].hist(df1['recall'], bins=30, alpha=0.5, label=args.name1, color='red')
    axes[1, 0].hist(df2['recall'], bins=30, alpha=0.5, label=args.name2, color='blue')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Recall Distribution')
    axes[1, 0].legend()

    # Scatter plot: Model1 vs Model2 F1
    axes[1, 1].scatter(df_combined['f1_model1'], df_combined['f1_model2'], alpha=0.5, s=10)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal line
    axes[1, 1].set_xlabel(f'{args.name1} F1')
    axes[1, 1].set_ylabel(f'{args.name2} F1')
    axes[1, 1].set_title('Per-Image F1 Comparison')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plots_file = output_dir / 'comparison_plots.png'
    plt.savefig(plots_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plots saved to: {plots_file}")

    # Summary report
    summary = {
        'comparison': {
            'model1': {
                'name': args.name1,
                'path': args.model1,
                'avg_precision': float(df1['precision'].mean()),
                'avg_recall': float(df1['recall'].mean()),
                'avg_f1': float(df1['f1'].mean()),
                'total_tp': int(df1['tp'].sum()),
                'total_fp': int(df1['fp'].sum()),
                'total_fn': int(df1['fn'].sum()),
            },
            'model2': {
                'name': args.name2,
                'path': args.model2,
                'avg_precision': float(df2['precision'].mean()),
                'avg_recall': float(df2['recall'].mean()),
                'avg_f1': float(df2['f1'].mean()),
                'total_tp': int(df2['tp'].sum()),
                'total_fp': int(df2['fp'].sum()),
                'total_fn': int(df2['fn'].sum()),
            },
            'winner': args.name1 if df1['f1'].mean() > df2['f1'].mean() else args.name2,
            'f1_difference': abs(float(df1['f1'].mean() - df2['f1'].mean())),
        }
    }

    summary_file = output_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"Winner (by F1): {summary['comparison']['winner']}")
    print(f"F1 Difference:  {summary['comparison']['f1_difference']:.4f}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Metrics:        {metrics_file}")
    print(f"  - Visualizations: {viz_dir}")
    print(f"  - Plots:          {plots_file}")
    print(f"  - Summary:        {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
