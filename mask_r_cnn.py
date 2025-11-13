import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms import functional as F


class FootballCocoDataset(Dataset):
    """
    Minimal COCO-style dataset loader that returns torchvision-friendly targets.
    """

    def __init__(
        self,
        images_root: Path,
        annotation_file: Path,
        transforms: Optional["Compose"] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms

        with open(self.annotation_file, "r") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda im: im["id"])
        self.annotations = coco["annotations"]
        self.categories = sorted(coco["categories"], key=lambda c: c["id"])

        self.cat_id_to_contig: Dict[int, int] = {
            cat["id"]: idx + 1 for idx, cat in enumerate(self.categories)
        }
        self.contig_to_cat_id: Dict[int, int] = {
            contig: cat_id for cat_id, contig in self.cat_id_to_contig.items()
        }

        self.ann_index = defaultdict(list)
        for ann in self.annotations:
            self.ann_index[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def num_classes(self) -> int:
        return len(self.cat_id_to_contig) + 1  # + background

    def to_coco_label(self, label: int) -> int:
        return self.contig_to_cat_id.get(int(label), 0)

    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        image_id = image_info["id"]
        image_path = self.images_root / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")

        annos = self.ann_index.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annos:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contig[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)

        if labels:
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(image_id),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class Compose:
    """Compose transforms that expect (image, target)."""

    def __init__(self, transforms: Iterable):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = torch.flip(image, dims=[2])
            width = image.shape[2]
            boxes = target["boxes"]
            if boxes.numel() > 0:
                xmin = width - boxes[:, 2]
                xmax = width - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes
        return image, target


def get_transforms(train: bool) -> Compose:
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    if use_pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value

        if (step + 1) % print_freq == 0:
            print(
                f"[Epoch {epoch}] Step {step + 1}/{len(data_loader)} "
                f"Loss: {loss_value:.4f}"
            )

    return running_loss / max(len(data_loader), 1)


def coco_evaluate(annotation_file: Path, predictions: List[Dict]) -> Dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO evaluation. "
            "Install it before running evaluation."
        ) from exc

    coco_gt = COCO(str(annotation_file))
    if not predictions:
        return {"mAP_50_95": 0.0, "mAP_50": 0.0}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {"mAP_50_95": float(stats[0]), "mAP_50": float(stats[1])}


def compute_precision_recall(
    gt_records: List[Dict],
    pred_records: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    gt_by_image = {item["image_id"]: item for item in gt_records}
    pred_by_image = {item["image_id"]: item for item in pred_records}

    total_gt = torch.zeros(num_classes, dtype=torch.float64)
    true_positives = torch.zeros(num_classes, dtype=torch.float64)
    false_positives = torch.zeros(num_classes, dtype=torch.float64)

    for image_id, gt in gt_by_image.items():
        preds = pred_by_image.get(
            image_id,
            {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,)),
            },
        )

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = preds["boxes"]
        pred_labels = preds["labels"]
        pred_scores = preds["scores"]

        for class_id in range(1, num_classes):
            gt_mask = gt_labels == class_id
            pred_mask = pred_labels == class_id
            gt_cls = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]
            scores_cls = pred_scores[pred_mask]

            total_gt[class_id] += gt_cls.shape[0]

            if pred_cls.numel() == 0:
                continue
            if gt_cls.numel() == 0:
                false_positives[class_id] += pred_cls.shape[0]
                continue

            order = torch.argsort(scores_cls, descending=True)
            pred_cls = pred_cls[order]
            ious = box_iou(pred_cls, gt_cls)
            matched = set()

            for pred_idx in range(pred_cls.shape[0]):
                if ious.shape[1] == 0:
                    false_positives[class_id] += 1
                    continue
                best_gt = torch.argmax(ious[pred_idx]).item()
                best_iou = ious[pred_idx, best_gt].item()
                if best_iou >= iou_threshold and best_gt not in matched:
                    matched.add(best_gt)
                    true_positives[class_id] += 1
                else:
                    false_positives[class_id] += 1

    epsilon = 1e-6
    precision = float(
        true_positives.sum().item()
        / max(true_positives.sum().item() + false_positives.sum().item(), epsilon)
    )
    recall = float(
        true_positives.sum().item() / max(total_gt.sum().item(), epsilon)
    )
    return {"precision": precision, "recall": recall}


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    dataset: FootballCocoDataset,
    annotation_file: Path,
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    model.eval()
    coco_predictions: List[Dict] = []
    gt_records: List[Dict] = []
    pred_records: List[Dict] = []
    decode_label = dataset.contig_to_cat_id

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                gt_records.append(
                    {
                        "image_id": image_id,
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    }
                )

                boxes = output["boxes"].cpu()
                labels = output["labels"].cpu()
                scores = output["scores"].cpu()

                keep = scores >= score_threshold
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                positive = labels > 0
                boxes = boxes[positive]
                labels = labels[positive]
                scores = scores[positive]

                pred_records.append(
                    {
                        "image_id": image_id,
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                    }
                )

                for box, label, score in zip(boxes, labels, scores):
                    category_id = decode_label.get(int(label.item()))
                    if category_id is None or category_id == 0:
                        continue
                    coco_predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1]),
                            ],
                            "score": float(score.item()),
                        }
                    )

    metrics = compute_precision_recall(
        gt_records, pred_records, dataset.num_classes
    )
    coco_metrics = coco_evaluate(annotation_file, coco_predictions)
    metrics.update(coco_metrics)
    return metrics


def save_checkpoint(model: nn.Module, optimizer, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def load_checkpoint(model: nn.Module, optimizer, path: Path, device: torch.device) -> int:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask R-CNN training entrypoint")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--images-root", type=Path, default=Path("data-mask-r-cnn/images"))
    parser.add_argument("--train-ann", type=Path, default=Path("data-mask-r-cnn/annotations/train.json"))
    parser.add_argument("--val-ann", type=Path, default=Path("data-mask-r-cnn/annotations/val.json"))
    parser.add_argument("--eval-ann", type=Path, help="Annotation file for evaluation mode")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/maskrcnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-step-size", type=int, default=8)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume training")
    parser.add_argument("--eval-checkpoint", type=Path, help="Checkpoint for evaluation-only mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train_dataset = FootballCocoDataset(
            args.images_root,
            args.train_ann,
            transforms=get_transforms(train=True),
        )
        val_dataset = FootballCocoDataset(
            args.images_root,
            args.val_ann,
            transforms=get_transforms(train=False),
        )
        num_classes = train_dataset.num_classes

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        model = build_model(num_classes, use_pretrained=not args.no_pretrained)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )

        start_epoch = 0
        if args.resume:
            start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1
            print(f"Resumed from epoch {start_epoch}")

        best_map = 0.0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "metrics.jsonl"

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            avg_loss = train_one_epoch(
                model, optimizer, train_loader, device, epoch, args.print_freq
            )
            lr_scheduler.step()

            metrics = evaluate(
                model,
                val_loader,
                device,
                val_dataset,
                args.val_ann,
                score_threshold=args.score_threshold,
            )
            metrics["train_loss"] = avg_loss
            metrics["epoch"] = epoch
            metrics["epoch_time"] = time.time() - epoch_start

            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            print(
                f"Epoch {epoch}: loss={avg_loss:.4f}, "
                f"mAP@0.5:0.95={metrics['mAP_50_95']:.4f}, "
                f"mAP@0.5={metrics['mAP_50']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

            if metrics["mAP_50_95"] > best_map:
                best_map = metrics["mAP_50_95"]
                ckpt_path = args.output_dir / "best_model.pth"
                save_checkpoint(model, optimizer, epoch, ckpt_path)
                print(f"Saved new best checkpoint to {ckpt_path}")

    else:  # Evaluation-only mode
        if args.eval_ann is None:
            raise ValueError("Provide --eval-ann when mode=eval")
        if args.eval_checkpoint is None:
            raise ValueError("Provide --eval-checkpoint for evaluation mode")

        dataset = FootballCocoDataset(
            args.images_root,
            args.eval_ann,
            transforms=get_transforms(train=False),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        model = build_model(dataset.num_classes, use_pretrained=False)
        model.to(device)
        load_checkpoint(model, None, args.eval_checkpoint, device)
        metrics = evaluate(
            model,
            loader,
            device,
            dataset,
            args.eval_ann,
            score_threshold=args.score_threshold,
        )
        print(
            f"Evaluation results -> "
            f"mAP@0.5:0.95={metrics['mAP_50_95']:.4f}, "
            f"mAP@0.5={metrics['mAP_50']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
