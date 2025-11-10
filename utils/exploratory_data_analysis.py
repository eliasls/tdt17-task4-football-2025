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
        
        print("TAG:")
        print(root.tag)
        meta = root.findall("meta")
        print(meta)
        #task = meta.findall("task")
        #print(task.tag)
        
        
        orig = root.find("original_size")

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