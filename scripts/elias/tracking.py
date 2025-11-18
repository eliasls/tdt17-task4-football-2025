#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Track players and ball in football videos with Ultralytics")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11s_1920_rect_e150_best.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="/configs/botsort.yaml",
        help="Path to BoT-SORT tracker config",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/track/elias",
        help="Base directory for tracking outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Subfolder name for this tracking run",
    )
    parser.add_argument(
        "--save-predictions-json",
        type=str,
        default=None,
        help="Path to save tracking predictions as JSON",
    )
    parser.add_argument(
        "--save-predictions-mot",
        type=str,
        default=None,
        help="Path to save tracking predictions in MOT format",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FOOTBALL TRACKING (Ultralytics + BoT-SORT)")
    print("=" * 80)
    print(f"Input video:  {args.input}")
    print(f"Model:        {args.model}")
    print(f"Tracker cfg:  {args.tracker}")
    print(f"Project:      {args.project}")
    print(f"Name:         {args.name}")
    print("=" * 80)

    model = YOLO(args.model)

    all_predictions = []

    results = model.track(
        source=args.input,
        tracker=args.tracker,
        save=True,              
        project=args.project,
        name=args.name,
        stream=True,            
        verbose=False,
    )
    
    print("\nProcessing video and collecting tracks...")
    frame_count = 0
    
    for frame_count, r in enumerate(results, start=1):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        ids = boxes.id
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls

        if ids is None:
            continue

        ids = ids.cpu().numpy().astype(int)
        xyxy = xyxy.cpu().numpy()
        conf = conf.cpu().numpy()
        cls = cls.cpu().numpy().astype(int)

        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1

            all_predictions.append({
                "frame": frame_count,                  
                "track_id": int(ids[i]),
                "class": int(cls[i]),
                "confidence": float(conf[i]),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "bbox_xywh": [float(x1), float(y1), float(w), float(h)],
            })
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total predictions:      {len(all_predictions)}")

    if args.save_predictions_json and all_predictions:
        json_path = Path(args.save_predictions_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "video": args.input,
                        "model": args.model,
                        "tracker": args.tracker,
                        "total_frames": frame_count,
                    },
                    "predictions": all_predictions,
                },
                f,
                indent=2,
            )
        print(f"Predictions JSON saved to: {json_path}")

    if args.save_predictions_mot and all_predictions:
        mot_path = Path(args.save_predictions_mot)
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mot_path, "w") as f:
            for p in all_predictions:
                frame = p["frame"]
                tid = p["track_id"]
                x, y, w, h = p["bbox_xywh"]
                conf = p["confidence"]
                f.write(
                    f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                )
        print(f"Predictions in MOT format saved to: {mot_path}")

    print("\nTracking complete.")
    print("=" * 80)
    print(f"Annotated video is in: {Path(args.project) / args.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()