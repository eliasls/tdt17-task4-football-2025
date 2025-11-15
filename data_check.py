import importlib
import os, sys
import cv2

PROJECT_ROOT = "/cluster/work/eliasls/tdt17/task4"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
  
  


def load_yolo_labels(label_path):
    """
    Load YOLO-style labels from a .txt file.
    Each line: class_id x_center y_center width height (all normalized).
    Returns a list of (class_id, x_c, y_c, w, h).
    """
    boxes = []
    if not os.path.exists(label_path):
        print(f"Warning: label file not found: {label_path}")
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {label_path}: {line}")
                continue
            cls, x_c, y_c, w, h = parts
            boxes.append((
                int(cls),
                float(x_c),
                float(y_c),
                float(w),
                float(h),
            ))
    return boxes


def draw_boxes_on_image(image_path, label_path, output_path, box_color=(0, 255, 0), thickness=2):
    """
    Draw boxes from YOLO label_path onto image_path and save to output_path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    boxes = load_yolo_labels(label_path)

    for cls, x_c, y_c, bw, bh in boxes:
        # Convert normalized YOLO coords to pixel coords
        x_center = x_c * w
        y_center = y_c * h
        box_w = bw * w
        box_h = bh * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
        # Optional: draw class id
        cv2.putText(
            img,
            str(cls),
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )

    # Save as PNG
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    
if __name__ == "__main__":
  
    match = "RBK-HamKam"
    frames = ["frame_000892.png", "frame_001067.png", "frame_1072.png", "frame_001157.png", "frame_001166.png"]

    frame = "frame_000892"
    frame_id = "000892"

    #har
    #frame_1 = label_1

    #skal ha
    #frame 1 = label_0

    image = "frame_000892.png"
    label_used = "frame_000892.txt"
    label_true = "frame_000891.txt"

    hamkam_image_path = "/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/data/images/train"
    hamkam_labels_path = "/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-HamKam/labels/train"

    image_path = hamkam_image_path + "/" + image
    label_used_path = hamkam_labels_path + "/" + label_used
    label_used_true = hamkam_labels_path + "/" + label_true


    # Output PNGs
    out_pred = "image_with_used_boxes.png"
    out_true = "image_with_true_boxes.png"

    # Draw predicted/“used” labels (e.g. possibly wrong mapping)
    draw_boxes_on_image(
        image_path=image_path,
        label_path=label_used_path,
        output_path=out_pred,
        box_color=(0, 0, 255),  # red boxes
    )

    # Draw ground-truth labels (the “true” mapping)
    draw_boxes_on_image(
        image_path=image_path,
        label_path=label_used_true,
        output_path=out_true,
        box_color=(0, 255, 0),  # green boxes
    )
    
print(os.getcwd())
