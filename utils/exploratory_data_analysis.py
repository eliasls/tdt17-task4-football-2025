import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


def print_lines_in_file(file_path, number_of_lines):
  print("hade")
  with open(file_path, "r") as f:
    lines = f.readlines()

  for line in lines[:number_of_lines]:
      print(line.strip())
      
def check_corrupts_and_artifacts(images, img_dir):
  bad_files = []
  black_ish_images = []
  for name in images:
    path = os.path.join(img_dir, name)
    img = cv2.imread(path)
    if img is None:
        bad_files.append(path)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    if mean_val < 5:   # super dark
        black_ish_images.append(name)
  return bad_files, black_ish_images

def get_missing_labels(images, labels_dir):
  missing_labels = []
  for name in images:
      stem, _ = os.path.splitext(name)
      label_path = os.path.join(labels_dir, stem + ".txt")
      if not os.path.exists(label_path):
          missing_labels.append(stem)
  return missing_labels

  return missing_labels

def generate_paths(root_dir, match_names):
    all_paths = {}

    for match in match_names:
        match_path = os.path.join(root_dir, match)

        if match == "RBK-BODO":
            parts = sorted(os.listdir(match_path))
            for part in parts:
                part_dir = os.path.join(match_path, part)

                inner_names = sorted(os.listdir(part_dir))
                inner_root = os.path.join(part_dir, inner_names[0])

                key_name = f"RBK-BODO-{part.upper()}" 

                all_paths[key_name] = {
                    "images": os.path.join(inner_root, "data", "images", "train"),
                    "labels": os.path.join(inner_root, "labels", "train"),
                    "annotations": os.path.join(inner_root, "annotations.xml"),
                    "data_yaml": os.path.join(inner_root, "data.yaml"),
                }

        else:
            # normal matches
            all_paths[match] = {
                "images": os.path.join(match_path, "data", "images", "train"),
                "labels": os.path.join(match_path, "labels", "train"),
                "annotations": os.path.join(match_path, "annotations.xml"),
                "data_yaml": os.path.join(match_path, "data.yaml"),
            }

    return all_paths




def check_original_sizes(paths_dict, expected_width=1920, expected_height=1080):
    results = {}

    for name, paths in paths_dict.items():
        xml_path = paths.get("annotations")
        entry = {
            "status": None,
            "width": None,
            "height": None,
            "path": xml_path,
        }

        if not xml_path or not os.path.exists(xml_path):
            entry["status"] = "missing_xml"
            results[name] = entry
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        meta_el = root.find("meta")
        if meta_el is None:
            entry["status"] = "missing_meta"
            results[name] = entry
            continue

        # try 1) /meta/task/original_size
        orig = None
        task_el = meta_el.find("task")
        if task_el is not None:
            orig = task_el.find("original_size")

        # if not found, try 2) /meta/original_size (your RBK-BODO-PART1/2 case)
        if orig is None:
            orig = meta_el.find("original_size")

        if orig is None:
            entry["status"] = "missing_original_size"
            results[name] = entry
            continue

        w_el = orig.find("width")
        h_el = orig.find("height")
        w = w_el.text if w_el is not None else None
        h = h_el.text if h_el is not None else None

        entry["width"] = w
        entry["height"] = h

        if str(w) == str(expected_width) and str(h) == str(expected_height):
            entry["status"] = "ok"
        else:
            entry["status"] = "mismatch"

        results[name] = entry

    return results



def sample_image_sizes(match_name, paths_dict, n=5):
    img_dir = paths_dict[match_name]["images"]
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    sizes = []
    for f in files[:n]:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            sizes.append((f, None, None))
        else:
            h, w = img.shape[:2]
            sizes.append((f, w, h))
    return sizes

def load_cvat_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta = root.find("meta")
    if meta is None:
        return None, None, None

    task = meta.find("task")
    if task is not None:
        orig = task.find("original_size")
        if orig is not None:
            w = int(orig.find("width").text)
            h = int(orig.find("height").text)
            return w, h, meta

    orig = meta.find("original_size")
    if orig is not None:
        w = int(orig.find("width").text)
        h = int(orig.find("height").text)
        return w, h, meta

    return None, None, meta

def collect_bbox_areas(xml_path):
    w, h, _ = load_cvat_metadata(xml_path)
    if w is None or h is None:
        raise ValueError(f"Could not find original_size in {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rel_areas = []   
    abs_areas = []   

    for track in root.findall("track"):
        for box in track.findall("box"):
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            bw = xbr - xtl  
            bh = ybr - ytl  

            abs_area = bw * bh                    
            rel_area = abs_area / (w * h)         

            abs_areas.append(abs_area)
            rel_areas.append(rel_area)

    return rel_areas, abs_areas

def load_cvat_track(xml_path):
    
    print(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tracks = root.findall("track")
    track1 = tracks[0]
    track1.get
    player = 0
    referee = 0
    ball = 0
    event_labels = 0
    
    #attri = tracks[0].find("box").find("./attribute[@name='team']")
    #print(attri.text)
    
    for track in tracks:
        if track.get("label") == "ball":
            boxes = track.findall("box")
            ball += len(boxes)
        elif track.get("label") == "event_labels" or track.get("label") == "event":
            boxes = track.findall("box")
            event_labels += len(boxes)
        else:
            boxes = track.findall("box")
            for box in boxes:
                #print(track.get("label"))
                if box.find("./attribute[@name='team']").text == "referee":
                    referee += 1
                else: 
                    player += 1
    
    return player, referee, ball, event_labels

#def collect_class_stats(xml_path):
    