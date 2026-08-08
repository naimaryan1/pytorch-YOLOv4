# -*- coding: utf-8 -*-

import argparse
import cv2
import math
import os
import random
import tempfile

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

CPU example:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 1
  -output_inference

Single GPU:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch150.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -gpu-cores 0

Multiple GPUs:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch150.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -gpu-cores 0,1
  
  
"""
# -*- coding: utf-8 -*-

import argparse
import cv2
import math
import os
import random
import tempfile

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


SUBSET_SEED = 40
INFERENCE_SCORE_THRESHOLD = 0.40


def create_test_subset(test_label_path, coverage_ratio):
    if not (0.0 < coverage_ratio <= 1.0):
        raise ValueError(
            "test_coverage_ratio must satisfy 0 < ratio <= 1"
        )

    with open(test_label_path, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    total_files = len(lines)

    if total_files == 0:
        raise RuntimeError(
            f"No test entries found in: {test_label_path}"
        )

    if coverage_ratio == 1.0:
        selected_lines = lines
    else:
        number_to_test = max(
            1,
            int(math.ceil(total_files * coverage_ratio))
        )

        rng = random.Random(SUBSET_SEED)

        selected_indices = sorted(
            rng.sample(
                range(total_files),
                number_to_test
            )
        )

        selected_lines = [
            lines[i]
            for i in selected_indices
        ]

    image_paths = [
        line.split()[0]
        for line in selected_lines
    ]

    subset_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="yolov4_test_subset_",
        delete=False,
        encoding="utf-8"
    )

    with subset_file:
        subset_file.write(
            "\n".join(selected_lines) + "\n"
        )

    return (
        subset_file.name,
        image_paths,
        total_files,
        len(selected_lines)
    )


def save_inference_image(
    image,
    boxes_xywh,
    scores,
    labels,
    source_path,
    dataset_dir,
    results_dir
):
    output_image = image.copy()

    for box, score, label in zip(
        boxes_xywh,
        scores,
        labels
    ):
        score_value = float(score)

        if score_value < INFERENCE_SCORE_THRESHOLD:
            continue

        x, y, w, h = [
            float(value)
            for value in box
        ]

        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = (
            f"class {int(label)} "
            f"{score_value:.2f}"
        )

        cv2.putText(
            output_image,
            text,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    image_name = os.path.basename(source_path)

    output_path = os.path.join(
        results_dir,
        image_name
    )

    # Dataset images are RGB in the validation path.
    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            output_image,
            cv2.COLOR_RGB2BGR
        )
    )


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    cfg,
    device,
    image_paths,
    output_inference=False
):
    model.eval()

    coco = convert_to_coco_api(
        data_loader.dataset,
        bbox_fmt="coco"
    )

    coco_evaluator = CocoEvaluator(
        coco,
        iou_types=["bbox"],
        bbox_fmt="coco"
    )

    results_dir = None

    if output_inference:
        results_dir = os.path.join(
            cfg.dataset_dir,
            "results"
        )

        os.makedirs(
            results_dir,
            exist_ok=True
        )

        print(
            f"Inference output directory: "
            f"{results_dir}"
        )

    image_index = 0

    for images, targets in data_loader:
        model_input = [
            [cv2.resize(img, (cfg.w, cfg.h))]
            for img in images
        ]

        model_input = np.concatenate(
            model_input,
            axis=0
        )

        model_input = model_input.transpose(
            0, 3, 1, 2
        )

        model_input = torch.from_numpy(
            model_input
        ).div(255.0)

        model_input = model_input.to(
            device=device,
            dtype=torch.float32
        )

        targets = [
            {
                k: v.to(device)
                for k, v in target.items()
            }
            for target in targets
        ]

        outputs = model(model_input)

        res = {}

        for img, target, boxes, confs in zip(
            images,
            targets,
            outputs[0],
            outputs[1]
        ):
            img_height, img_width = img.shape[:2]

            boxes = boxes.squeeze(2).cpu().numpy()

            # normalized [x1,y1,x2,y2]
            # -> COCO [x,y,w,h]
            boxes[..., 2:] = (
                boxes[..., 2:] -
                boxes[..., :2]
            )

            boxes[..., 0] *= img_width
            boxes[..., 1] *= img_height
            boxes[..., 2] *= img_width
            boxes[..., 3] *= img_height

            confs = confs.cpu().numpy()

            labels = np.argmax(
                confs,
                axis=1
            ).flatten()

            scores = np.max(
                confs,
                axis=1
            ).flatten()

            boxes_tensor = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels_tensor = torch.as_tensor(
                labels,
                dtype=torch.int64
            )

            scores_tensor = torch.as_tensor(
                scores,
                dtype=torch.float32
            )

            res[target["image_id"].item()] = {
                "boxes": boxes_tensor,
                "scores": scores_tensor,
                "labels": labels_tensor,
            }

            if output_inference:
                save_inference_image(
                    image=img,
                    boxes_xywh=boxes,
                    scores=scores,
                    labels=labels,
                    source_path=image_paths[
                        image_index
                    ],
                    dataset_dir=cfg.dataset_dir,
                    results_dir=results_dir
                )

            image_index += 1

        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLOv4 checkpoint "
            "on custom test.txt"
        )
    )

    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        required=True,
        type=str,
        help="Path to Yolov4_epochXXX.pth"
    )

    parser.add_argument(
        "-test_label",
        "--test_label",
        required=True,
        type=str,
        help="Path to test.txt"
    )

    parser.add_argument(
        "-dir",
        "--data-dir",
        required=True,
        dest="dataset_dir",
        type=str,
        help="Dataset root directory"
    )

    parser.add_argument(
        "-classes",
        "--classes",
        required=True,
        type=int,
        help="Number of classes"
    )

    parser.add_argument(
        "-test_coverage_ratio",
        "--test_coverage_ratio",
        default=1.0,
        type=float,
        help=(
            "Fraction of test.txt to evaluate. "
            "Must satisfy 0 < ratio <= 1. "
            "Default: 1"
        )
    )

    parser.add_argument(
        "-output_inference",
        "--output_inference",
        action="store_true",
        help=(
            "Save annotated inference images "
            "to <dir>/results"
        )
    )

    parser.add_argument(
        "-g",
        "--gpu",
        default="0",
        type=str,
        help="GPU index"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if not (
        0.0 <
        args.test_coverage_ratio <=
        1.0
    ):
        raise ValueError(
            "test_coverage_ratio must satisfy "
            "0 < ratio <= 1"
        )

    cfg = edict(dict(Cfg))
    cfg.dataset_dir = args.dataset_dir
    cfg.classes = args.classes

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        subset_label_path,
        test_image_paths,
        total_test_files,
        actual_test_files
    ) = create_test_subset(
        args.test_label,
        args.test_coverage_ratio
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test labels: {args.test_label}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Classes: {args.classes}")
    print(
        f"Test coverage ratio: "
        f"{args.test_coverage_ratio}"
    )
    print(
        f"Total files in test list: "
        f"{total_test_files}"
    )
    print(
        f"Files actually tested: "
        f"{actual_test_files}"
    )

    test_dataset = Yolo_dataset(
        subset_label_path,
        cfg,
        train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=(
            cfg.batch //
            cfg.subdivisions
        ),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate
    )

    model = Yolov4(
        None,
        n_classes=cfg.classes,
        inference=True
    )

    state_dict = torch.load(
        args.checkpoint,
        map_location=device
    )

    if any(
        key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.replace(
                "module.",
                "",
                1
            ): value
            for key, value
            in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Running COCO evaluation...")

    try:
        evaluator = evaluate(
            model=model,
            data_loader=test_loader,
            cfg=cfg,
            device=device,
            image_paths=test_image_paths,
            output_inference=(
                args.output_inference
            )
        )
    finally:
        if os.path.exists(
            subset_label_path
        ):
            os.remove(
                subset_label_path
            )

    stats = evaluator.coco_eval[
        "bbox"
    ].stats

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