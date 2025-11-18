import xml.etree.ElementTree as ET
from pathlib import Path


def xml_to_mot(xml_path: Path, mot_path: Path):
    print(f"Reading XML: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    for track in root.findall("track"):
        track_id = int(track.attrib["id"])
        label = track.attrib.get("label", "")

        if label not in {"player", "ball"}:
            continue

        for box in track.findall("box"):
            frame_xml = int(box.attrib["frame"])   
            frame = frame_xml + 1                  

            outside = int(box.attrib.get("outside", 0))
            if outside == 1:

                continue

            x1 = float(box.attrib["xtl"])
            y1 = float(box.attrib["ytl"])
            x2 = float(box.attrib["xbr"])
            y2 = float(box.attrib["ybr"])
            w = x2 - x1
            h = y2 - y1


            line = f"{frame},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
            lines.append(line)

    lines.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))

    mot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mot_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote MOT/HOTA GT to: {mot_path}")
    print(f"Total GT boxes: {len(lines)}")


def main():
    xml_path_string = "/cluster/projects/vc/courses/TDT17/other/Football2025/RBK-AALESUND/annotations.xml"
    mot_path_string = "/cluster/work/eliasls/tdt17/task4/gt-tracking/RBK-AALESUND/gt.txt"
    xml_path = Path(xml_path_string)
    mot_path = Path(mot_path_string)

    xml_to_mot(xml_path, mot_path)


if __name__ == "__main__":
    main()