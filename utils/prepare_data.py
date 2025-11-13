import shutil
from pathlib import Path


def copy_file(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    
    
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

CATEGORY_MAP = {
    "player": 1,
    "referee": 2,
    "ball": 3,
}
CATEGORIES = [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "referee"},
    {"id": 3, "name": "ball"},
]

def parse_cvat_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            yield {
                "frame": frame,
                "bbox": [xtl, ytl, xbr, ybr],
                "class": label,
            }

def build_split(split_name, matches, paths, image_root, out_json):
    images, annotations = [], []
    image_lookup = {}
    image_id = 1
    ann_id = 1

    for match in matches:
        match_img_dir = Path(image_root, split_name, match)
        xml_path = Path(paths[match]["annotations"])
        for img_file in sorted(match_img_dir.glob("*.png")):
            frame_idx = int(img_file.stem.split("_")[-1])  # frame_000123.png -> 123
            with Image.open(img_file) as im:
                width, height = im.size
            images.append({
                "id": image_id,
                "file_name": str(img_file.relative_to(image_root)),
                "width": width,
                "height": height,
                "match": match,
                "frame_idx": frame_idx,
            })
            image_lookup[(match, frame_idx)] = image_id
            image_id += 1

        for box in parse_cvat_boxes(xml_path):
            if box["class"] == "event" or box["class"] == "event_labels":
              continue
            key = (match, box["frame"])
            if key not in image_lookup:
                continue  # frame not exported in this split
            x1, y1, x2, y2 = box["bbox"]
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": image_lookup[key],
                "category_id": CATEGORY_MAP[box["class"]],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": CATEGORIES},
            f,
        )
    print(f"Saved {out_json} with {len(images)} images, {len(annotations)} boxes")