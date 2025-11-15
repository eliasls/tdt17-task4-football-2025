import os, sys
import shutil
from pathlib import Path
PROJECT_ROOT = "/cluster/work/eliasls/tdt17/task4"
if PROJECT_ROOT not in sys.path:
  sys.path.append(PROJECT_ROOT)
from utils import exploratory_data_analysis
print(os.getcwd())


def zero_index_frames(path):
  dir = os.path.join(path, "")
  frames = sorted(os.listdir(dir))
  for frame in frames:
    old_frame_path = path + "/" + frame
    punc_split = frame.split(".")
    under_score_split = punc_split[0].split("_")
    old_frame_id = under_score_split[1]
    old_frame_id_as_int = int(old_frame_id)
    new_frame_id_as_int = old_frame_id_as_int - 1
    new_frame_id = f"{new_frame_id_as_int:06d}"
    new_frame_path = path + "/" + under_score_split[0] + "_" +  new_frame_id + "." + punc_split[1]
    os.rename(old_frame_path, new_frame_path)
  
print(os.getcwd())


def copy_file(src_path: str, dst_path: str) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

root_dir = os.path.join("/cluster/projects/vc/courses/TDT17/other/Football2025", "")
matches = sorted(os.listdir(root_dir))
paths = exploratory_data_analysis.generate_paths(root_dir, matches)

path1 = "data-mask-r-cnn/images/test/RBKAALESUND"
path2 = "data-mask-r-cnn/images/val/RBK-FREDRIKSTAD"

dst_path = Path("/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/labels/train/RBK-BODO-PART1")
src_path = Path("/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-BODO/part1/RBK_BODO_PART1/labels/train")
images = sorted(os.listdir(os.path.join(str(src_path), "")))

def fetch_folder(src_path, dst_path, images):
  for image in images:
    final = "RBK_HamKam_" + image
    copy_file(str(src_path / image), str(dst_path / final))
  #zero_index_frames(str(dst_path))


#fetch_folder(src_path, dst_path, images)


#base = dst_path = Path("/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/labels/train")

def get_labels(matches):
  for match in matches:
    print(match)
    labels = sorted(os.listdir(os.path.join(paths[match]["labels"], "")))
    dst_path = base
    src_path = Path(paths[match]["labels"])
    print(f"src: {src_path}")
    print(f"dst: {dst_path}")
    fetch_folder(src_path, dst_path, labels)
    
    print(f"done{match}")
      
#get_labels(train3)
#get_labels(val)

def rename(src_path, adding):
  elements = sorted(os.listdir(os.path.join(src_path, "")))
  count = 0
  for element in elements:
    old = src_path + "/" + element
    id = f"{count:06d}"
    new = src_path + "/" + adding + "frame_" + "id" + ".txt"
    os.rename(old, new)
    count += 1
    
def back(src_path):
  elements = sorted(os.listdir(os.path.join(src_path, "")))
  for element in elements:
    l = element.split("_")
    old = src_path + "/" + element
    new = src_path + l[-2] + "_" + l[-1]
    os.rename(old, new)

def get_images(matches):
  for match in matches:
    print(match)
    labels = sorted(os.listdir(os.path.join(paths[match]["images"], "")))
    dst_path = base / match 
    src_path = Path(paths[match]["images"])
    print(f"src: {src_path}")
    print(f"dst: {dst_path}")
    fetch_folder(src_path, dst_path, labels)
    
    print(f"done{match}")
    
def move_file(matches, src_path, dst_path):
  for match in matches:
    print(match)
    labels = sorted(os.listdir(os.path.join(src_path + "/" + match , "")))
    print(f"src: {src_path}")
    print(f"dst: {dst_path}")
    fetch_folder(src_path + "/" + match, dst_path, labels)
    print(f"done{match}")

#base = dst_path = Path("/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/images/train/")
train = ['RBK-BODO-PART1', 'RBK-BODO-PART2','RBK-BODO-PART3',"RBK-VIKING", "RBK-FREDRIKSTAD"]
test = ["RBK-AALESUND"]
val = ['RBK-HamKam']
#test = "RBK-BODO-PART1"
base = Path("/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/jon/val")

get_labels(val)

src = "/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/labels/val"
dst = "/cluster/work/eliasls/tdt17/task4/data-mask-r-cnn/labels/val"
#move_file(val, src, dst)
#get_images(train)

""" rename(str(base / train[0]),"RBK_BODO_PART1_")
rename(str(base / train[1]),"RBK_BODO_PART2_")
rename(str(base / train[2]),"RBK_BODO_PART3_")
rename(str(base / train[3]),"RBK_VIKING_")
rename(str(base / train[4]),"RBK-FREDRIKSTAD_") """

#rename(str(base / val[0]),"RBK_AALESUND_")


#back(str(base))
#rename(str(base),"RBK_HamKam_")


""" back(str(base / train[0]))
back(str(base / train[1]))
back(str(base / train[2]))
back(str(base / train[3]))
back(str(base / train[4])) """


#rename(str(base / test),"RBK_BODO_PART1_")

""" for match in train:
  rename(str(base / match), match + "_") """
    


def move_and_rename(src_path, dst_path, filetype):
  elements = sorted(os.listdir(os.path.join(src_path, "")))
  
