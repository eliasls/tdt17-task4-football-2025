#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_yolo_bbox_to_coco(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox to COCO format.

    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    COCO: [x_min, y_min, width, height] (absolute pixels)
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert from normalized to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert from center format to top-left corner format
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)

    return [x_min, y_min, width_abs, height_abs]


def convert_split(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    class_info: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_images_dir = output_dir / split_name
    output_images_dir.mkdir(parents=True, exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class_id,
                "name": info["name"],
                "supercategory": info["supercategory"]
            }
            for class_id, info in class_info.items()
        ],
    }

    annotation_id = 1
    image_id = 1

    image_files = sorted(images_dir.glob("*.png"))

    if not image_files:
        print(f"  WARNING: No images found in {images_dir}")
        return

    print(f"  Found {len(image_files)} images")

    for img_path in tqdm(image_files, desc=f"  Processing {split_name}"):
        label_path = labels_dir / f"{img_path.stem}.txt"

        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"  WARNING: Could not read image {img_path}: {e}")
            continue

        symlink_path = output_images_dir / img_path.name
        if not symlink_path.exists():
            if img_path.is_symlink():
                real_path = img_path.resolve()
                os.symlink(real_path, symlink_path)
            else:
                os.symlink(img_path.resolve(), symlink_path)

        image_data = {
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height,
        }
        coco_data["images"].append(image_data)

        if label_path.exists():
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    print(f"  WARNING: Invalid annotation in {label_path}: {line}")
                    continue

                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]

                coco_bbox = convert_yolo_bbox_to_coco(yolo_bbox, img_width, img_height)

                area = coco_bbox[2] * coco_bbox[3]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

        image_id += 1

    output_json = output_dir / split_name / "_annotations.coco.json"
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"  ✓ Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    print(f"  ✓ Output: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO format")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="football_dataset",
        help="Path to YOLO dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="football_dataset_coco",
        help="Path to output COCO dataset directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to convert (default: train val test)",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    class_info = {
        0: {"name": "player", "supercategory": "person"},
        1: {"name": "ball", "supercategory": "equipment"},
        2: {"name": "event_labels", "supercategory": "annotation"},
    }

    print(f"Classes:")
    for class_id, info in class_info.items():
        print(f"  {class_id}: {info['name']} (supercategory: {info['supercategory']})")
    print(f"Splits: {args.splits}")
    print("=" * 80)

    for split in args.splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        if not images_dir.exists():
            print(f"\nWARNING: Images directory not found: {images_dir}")
            continue

        if not labels_dir.exists():
            print(f"\nWARNING: Labels directory not found: {labels_dir}")
            continue

        convert_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            split_name=split,
            class_info=class_info,
        )

    for split in args.splits:
        split_dir = output_dir / split
        if split_dir.exists():
            json_file = split_dir / "_annotations.coco.json"
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                print(f"  {split}/")
                print(f"    _annotations.coco.json ({len(data['images'])} images, {len(data['annotations'])} annotations)")
                print(f"    <image symlinks>")
    print()


if __name__ == "__main__":
    main()
