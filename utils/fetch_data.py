import os, sys
import shutil
from pathlib import Path
PROJECT_ROOT = "/cluster/work/eliasls/tdt17/task4"
if PROJECT_ROOT not in sys.path:
  sys.path.append(PROJECT_ROOT)
from utils import exploratory_data_analysis
print(os.getcwd())

def copy_file(src_path, dst_path):
  src = Path(src_path)
  dst = Path(dst_path)
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)

def generate_dst_with_correct_name_img(dst_path, new_filename, old_filename, filetype):
  #frame_000002.txt
  #RBK_VIKING_frame_
  punc_split = old_filename.split(".")
  under_score_split = punc_split[0].split("_")
  
  new_frame_id_as_int = int(under_score_split[1]) - 1
  new_frame_id_as_string = f"{new_frame_id_as_int:06d}"
  dst_path_with_correct_name = dst_path + "/" + new_filename + new_frame_id_as_string + filetype
  return dst_path_with_correct_name

def generate_dst_with_correct_name_txt(dst_path, new_filename, old_filename, filetype):
  #frame_000002.png
  #RBK_VIKING_frame_
  old = old_filename.split("_")[1]
  dst_path_with_correct_name = dst_path + "/" + new_filename + old 
  return dst_path_with_correct_name
  
  

def move(src_path, dst_path, filename):
  files = sorted(os.listdir(src_path))
  #print(files)
  for file in files:
    if file.split(".")[1] == "txt":
      dst_path_with_correct_name = generate_dst_with_correct_name_txt(dst_path, filename, file, "." + file.split(".")[1])
      src_path_with_file = src_path + "/" + file
      #print("dst:" + dst_path_with_correct_name)
      #print("src" + src_path_with_file)
      #print("-----------------------------------")
      copy_file(src_path_with_file, dst_path_with_correct_name)
    elif file.split(".")[1] == "png":
      dst_path_with_correct_name = generate_dst_with_correct_name_img(dst_path, filename, file, "." + file.split(".")[1])
      src_path_with_file = src_path + "/" + file
      #print("dst:" + dst_path_with_correct_name)
      #print("src" + src_path_with_file)
      #print("-----------------------------------")
      copy_file(src_path_with_file, dst_path_with_correct_name)
  return


def get_data(move_dict):
  matches = move_dict.keys()
  for match in matches:
    for i in range(2):
      if i == 0:
        #img
        move(move_dict[match]["src_images"], move_dict[match]["dst_images"], move_dict[match]["file_name"])
      else:
        #label
        move(move_dict[match]["src_labels"], move_dict[match]["dst_labels"], move_dict[match]["file_name"])
  


def has_class(label_path: Path, class_id: int = 2):
    try:
        count = 0
        lines_read = 0
        with label_path.open("r") as f:
            for line in f:
                lines_read += 1
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls == class_id:
                    #print(label_path)
                    count += 1
                    #return True
            if count > 0:
                print(label_path)
                #print(lines_read)
            return count
    except FileNotFoundError:
        return 0
    return 0


def remove_class_from_labels(label_dir: Path, class_id: int = 2) -> None:
    """
    Remove lines starting with class_id from all .txt label files under label_dir.
    """
    for path in Path(label_dir).glob("*.txt"):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        kept = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed
            cls = int(float(parts[0]))
            if cls == class_id:
                continue  # drop this annotation
            kept.append(line)
        if len(kept) != len(lines):
            path.write_text("\n".join(kept) + ("\n" if kept else ""))
