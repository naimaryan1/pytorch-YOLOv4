# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from cfg import Cfg
from dataset import Yolo_dataset
from models import Yolov4
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator

"""
cd /content/gdrive/MyDrive/pytorch-YOLOv4

python evaluate_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch150.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -g 0
"""

@torch.no_grad()
def evaluate(model, data_loader, cfg, device):
    model.eval()

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt="coco")
    coco_evaluator = CocoEvaluator(
        coco,
        iou_types=["bbox"],
        bbox_fmt="coco"
    )

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device=device, dtype=torch.float32)

        targets = [
            {k: v.to(device) for k, v in target.items()}
            for target in targets
        ]

        outputs = model(model_input)

        res = {}

        for img, target, boxes, confs in zip(
            images, targets, outputs[0], outputs[1]
        ):
            img_height, img_width = img.shape[:2]

            boxes = boxes.squeeze(2).cpu().numpy()

            # normalized [x1,y1,x2,y2] -> COCO [x,y,w,h]
            boxes[..., 2:] = boxes[..., 2:] - boxes[..., :2]
            boxes[..., 0] *= img_width
            boxes[..., 1] *= img_height
            boxes[..., 2] *= img_width
            boxes[..., 3] *= img_height

            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            confs = confs.cpu().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            scores = np.max(confs, axis=1).flatten()

            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = torch.as_tensor(scores, dtype=torch.float32)

            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }

        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv4 checkpoint on custom test.txt"
    )

    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        required=True,
        type=str
    )

    parser.add_argument(
        "-test_label",
        "--test_label",
        required=True,
        type=str
    )

    parser.add_argument(
        "-dir",
        "--data-dir",
        required=True,
        dest="dataset_dir",
        type=str
    )

    parser.add_argument(
        "-classes",
        "--classes",
        required=True,
        type=int
    )

    parser.add_argument(
        "-g",
        "--gpu",
        default="0",
        type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    cfg = edict(dict(Cfg))
    cfg.dataset_dir = args.dataset_dir
    cfg.classes = args.classes

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test labels: {args.test_label}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Classes: {args.classes}")

    test_dataset = Yolo_dataset(
        args.test_label,
        cfg,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch // cfg.subdivisions,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate
    )

    # Same inference model path used by train.py validation.
    model = Yolov4(
        None,
        n_classes=cfg.classes,
        inference=True
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device
    )

    # Optional compatibility for DataParallel checkpoints.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Test images: {len(test_dataset)}")
    print("Running COCO evaluation...")

    evaluator = evaluate(
        model,
        test_loader,
        cfg,
        device
    )

    stats = evaluator.coco_eval["bbox"].stats

    print("\nCOCO metrics")
    print(f"AP      : {stats[0]:.3f}")
    print(f"AP50    : {stats[1]:.3f}")
    print(f"AP75    : {stats[2]:.3f}")
    print(f"AP small: {stats[3]:.3f}")
    print(f"AP med  : {stats[4]:.3f}")
    print(f"AP large: {stats[5]:.3f}")
    print(f"AR1     : {stats[6]:.3f}")
    print(f"AR10    : {stats[7]:.3f}")
    print(f"AR100   : {stats[8]:.3f}")
    print(f"AR small: {stats[9]:.3f}")
    print(f"AR med  : {stats[10]:.3f}")
    print(f"AR large: {stats[11]:.3f}")
