#!/usr/bin/env python3
"""
Simple video tracking script for football players and ball.

Tracks objects in match videos using trained YOLO model and SORT tracker.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import supervision as sv
from boxmot import ByteTrack  # Changed: boxmot instead of trackers
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Track players and ball in football videos")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/train/emil/new_data/yolo11n_1920/weights/best.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Path to save tracking predictions as JSON (optional)",
    )

    args = parser.parse_args()

    print("="*80)
    print("FOOTBALL TRACKING")
    print("="*80)
    print(f"Input video:  {args.input}")
    print(f"Output video: {args.output}")
    print(f"Model:        {args.model}")
    print()

    # Initialize tracker (BYTETracker is better than SORT)
    tracker = ByteTrack(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    )
    
    # Load YOLO model
    model = YOLO(args.model)
    
    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

    # Storage for predictions (for HOTA evaluation)
    all_predictions = []
    frame_count = 0

    def callback(frame, frame_idx):
        nonlocal frame_count
        frame_count = frame_idx + 1
        
        if frame_count % 100 == 0:
            print(f"  Processing frame {frame_count}...")
        
        # Run YOLO detection
        result = model(frame, imgsz=1920, conf=0.25, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Convert to boxmot format: [x1, y1, x2, y2, conf, class]
        if len(detections) > 0:
            dets_array = np.column_stack([
                detections.xyxy,
                detections.confidence,
                detections.class_id
            ])
        else:
            dets_array = np.empty((0, 6))
        
        # Apply tracking
        tracks = tracker.update(dets_array, frame)
        
        # Convert back to supervision format
        if len(tracks) > 0:
            detections = sv.Detections(
                xyxy=tracks[:, :4],
                confidence=tracks[:, 4],
                class_id=tracks[:, 5].astype(int),
                tracker_id=tracks[:, 6].astype(int)
            )
            
            # Save predictions if requested
            if args.save_predictions:
                for i in range(len(detections)):
                    all_predictions.append({
                        'frame': frame_count,
                        'track_id': int(detections.tracker_id[i]),
                        'class': int(detections.class_id[i]),
                        'confidence': float(detections.confidence[i]),
                        'bbox': detections.xyxy[i].tolist(),
                    })
        else:
            detections = sv.Detections.empty()
        
        # Annotate frame
        labels = [f"ID:{tid}" for tid in detections.tracker_id] if len(detections) > 0 else []
        annotated_frame = box_annotator.annotate(frame.copy(), detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
        
        return annotated_frame

    # Process video
    print("\nProcessing video...")
    sv.process_video(
        source_path=args.input,
        target_path=args.output,
        callback=callback,
    )

    print(f"\n✓ Done! Tracked video saved to: {args.output}")
    print(f"Total frames processed: {frame_count}")
    
    if all_predictions:
        unique_tracks = len(set(p['track_id'] for p in all_predictions))
        print(f"Unique tracks: {unique_tracks}")
        
        # Save predictions
        if args.save_predictions:
            pred_file = Path(args.save_predictions)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(pred_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'video': args.input,
                        'model': args.model,
                        'tracker': 'BYTETracker',
                        'total_frames': frame_count,
                        'unique_tracks': unique_tracks,
                    },
                    'predictions': all_predictions
                }, f, indent=2)
            
            print(f"Predictions saved to: {args.save_predictions}")
    
    print("="*80)


if __name__ == "__main__":
    main()