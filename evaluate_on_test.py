# -*- coding: utf-8 -*-
#Download evaluate_on_test_latest_20260808_v6_commands.py 5:37 8/8/2026

"""
8/8/2026

LATEST COMMAND EXAMPLES

1) CPU - defaults:
   confidence_threshold = 0.40
   nms_iou_threshold = 0.60

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 1


2) CPU - defaults + save inference images:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 0.1 \
  -output_inference


3) CPU - custom confidence/NMS:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 0.1 \
  -confidence_threshold 0.30 \
  -nms_iou_threshold 0.50 \
  -output_inference


4) Single GPU 0 - defaults:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 1 \
  -gpu-cores 0


5) Single GPU 0 - custom confidence/NMS + save inference images:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 0.1 \
  -confidence_threshold 0.30 \
  -nms_iou_threshold 0.50 \
  -output_inference \
  -gpu-cores 0


6) Multiple GPUs:

python evaluate_on_test.py \
  -checkpoint /content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth \
  -test_label /content/gdrive/MyDrive/pytorch-YOLOv4/data/test.txt \
  -dir /content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire \
  -classes 2 \
  -test_coverage_ratio 1 \
  -gpu-cores 0,1

Notes:
- CPU is used unless -gpu-cores is supplied.
- Default confidence threshold is 0.40.
- Default NMS IoU threshold is 0.60.
- Evaluation processes one image at a time.
- results/test_results.txt is always created.
- -output_inference also saves annotated images under results/.
"""

import argparse
import cv2
import math
import os
import random
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict

from cfg import Cfg
from dataset import Yolo_dataset
from models import Yolov4
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tool.utils import post_processing


SUBSET_SEED = 40

# Same default detection/NMS thresholds used by this repo's demo.py:
# do_detect(..., conf_thresh=0.4, nms_thresh=0.6, ...)
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
DEFAULT_NMS_IOU_THRESHOLD = 0.60


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
        selected_lines,
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
    results_dir,
    confidence_threshold
):
    output_image = image.copy()

    for box, score, label in zip(
        boxes_xywh,
        scores,
        labels
    ):
        score_value = float(score)

        if score_value < confidence_threshold:
            continue

        # Normalize possible model output shapes such as
        # [4], [4, 1], or [1, 4] to four scalar values.
        box = np.asarray(
            box,
            dtype=np.float32
        ).reshape(-1)

        if box.size != 4:
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
    truth_lines,
    confidence_threshold,
    nms_iou_threshold,
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

    results_dir = os.path.join(
        cfg.dataset_dir,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    test_results_path = os.path.join(
        results_dir,
        "test_results.txt"
    )

    print(
        f"Test results file: "
        f"{test_results_path}"
    )

    if output_inference:
        print(
            f"Inference output directory: "
            f"{results_dir}"
        )

    test_results_file = open(
        test_results_path,
        "w",
        encoding="utf-8"
    )

    inference_times_seconds = []
    image_index = 0

    progress = tqdm(
        data_loader,
        total=len(data_loader),
        desc="Test inference",
        unit="image"
    )

    for images, targets in progress:
        # DataLoader uses batch_size=1 so timing is true per-image
        # model-forward inference time.
        img = images[0]
        target = targets[0]

        model_input = cv2.resize(
            img,
            (cfg.w, cfg.h)
        )

        model_input = model_input.transpose(
            2, 0, 1
        )

        model_input = (
            torch.from_numpy(model_input)
            .unsqueeze(0)
            .div(255.0)
            .to(
                device=device,
                dtype=torch.float32
            )
        )

        target = {
            k: v.to(device)
            for k, v in target.items()
        }

        # CUDA operations are asynchronous, so synchronize immediately
        # before and after model inference for accurate timing.
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        inference_start = time.perf_counter()

        outputs = model(model_input)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        inference_end = time.perf_counter()

        inference_seconds = (
            inference_end -
            inference_start
        )

        inference_times_seconds.append(
            inference_seconds
        )

        progress.set_postfix(
            inference_ms=(
                f"{inference_seconds * 1000.0:.2f}"
            )
        )

        # Use the repository's existing YOLOv4 post-processing exactly.
        # It performs:
        #   1) class-score max/argmax
        #   2) confidence filtering
        #   3) class-aware nms_cpu()
        #
        # Model output boxes are normalized [x1, y1, x2, y2].
        # post_processing() returns each kept detection as:
        # [x1, y1, x2, y2, conf, conf, class_id]
        postprocessed_batch = post_processing(
            img,
            confidence_threshold,
            nms_iou_threshold,
            outputs
        )

        detections = (
            postprocessed_batch[0]
            if len(postprocessed_batch) > 0
            else []
        )

        img_height, img_width = img.shape[:2]

        if len(detections) > 0:
            detections = np.asarray(
                detections,
                dtype=np.float32
            ).reshape(-1, 7)

            # Still normalized coordinates here.
            boxes_xyxy = detections[:, 0:4].copy()
            scores = detections[:, 5].astype(
                np.float32
            )
            labels = detections[:, 6].astype(
                np.int64
            )

            # Convert post-NMS normalized [x1,y1,x2,y2]
            # to ABSOLUTE-PIXEL COCO [x,y,w,h].
            #
            # Ground-truth train/val/test lists use absolute pixel
            # coordinates, so predictions are converted to absolute
            # pixels before passing to CocoEvaluator.
            boxes = np.empty(
                (boxes_xyxy.shape[0], 4),
                dtype=np.float32
            )

            boxes[:, 0] = (
                boxes_xyxy[:, 0] *
                img_width
            )
            boxes[:, 1] = (
                boxes_xyxy[:, 1] *
                img_height
            )
            boxes[:, 2] = (
                boxes_xyxy[:, 2] -
                boxes_xyxy[:, 0]
            ) * img_width
            boxes[:, 3] = (
                boxes_xyxy[:, 3] -
                boxes_xyxy[:, 1]
            ) * img_height
        else:
            boxes = np.empty(
                (0, 4),
                dtype=np.float32
            )

            scores = np.empty(
                (0,),
                dtype=np.float32
            )

            labels = np.empty(
                (0,),
                dtype=np.int64
            )

        # IMPORTANT:
        # This repository's custom CocoEvaluator/loadRes() expects
        # each bbox in nested form [[x, y, w, h]], i.e. tensor
        # shape [N, 1, 4].  Keep a separate flat [N, 4] copy for
        # drawing/saving inference images.
        boxes_for_drawing = boxes.reshape(-1, 4)

        boxes_for_coco = boxes.reshape(-1, 1, 4)

        boxes_tensor = torch.as_tensor(
            boxes_for_coco,
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

        res = {
            target["image_id"].item(): {
                "boxes": boxes_tensor,
                "scores": scores_tensor,
                "labels": labels_tensor,
            }
        }

        inferred_parts = []

        if len(detections) > 0:
            for detection in detections:
                x1 = int(round(
                    float(detection[0]) * img_width
                ))
                y1 = int(round(
                    float(detection[1]) * img_height
                ))
                x2 = int(round(
                    float(detection[2]) * img_width
                ))
                y2 = int(round(
                    float(detection[3]) * img_height
                ))
                class_id = int(detection[6])

                inferred_parts.append(
                    f"{x1},{y1},{x2},{y2},{class_id}"
                )

        test_results_file.write(
            f"TRUTH:{truth_lines[image_index]} "
            f"INFERED:{' '.join(inferred_parts)}\n"
        )

        coco_evaluator.update(res)

        # Saving is intentionally outside the inference timer.
        if output_inference:
            save_inference_image(
                image=img,
                boxes_xywh=boxes_for_drawing,
                scores=scores,
                labels=labels,
                source_path=image_paths[
                    image_index
                ],
                dataset_dir=cfg.dataset_dir,
                results_dir=results_dir,
                confidence_threshold=confidence_threshold
            )

        image_index += 1

    test_results_file.close()

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if len(inference_times_seconds) == 0:
        raise RuntimeError(
            "No inference timings were recorded."
        )

    average_inference_seconds = (
        sum(inference_times_seconds) /
        len(inference_times_seconds)
    )

    average_inference_ms = (
        average_inference_seconds *
        1000.0
    )

    print("\nInference timing")
    print(
        f"Images timed             : "
        f"{len(inference_times_seconds)}"
    )
    print(
        f"Average inference time   : "
        f"{average_inference_ms:.3f} ms/image"
    )
    print(
        f"Average inference time   : "
        f"{average_inference_seconds:.6f} sec/image"
    )

    return (
        coco_evaluator,
        average_inference_seconds
    )


def select_device(gpu_cores):
    if gpu_cores is None:
        print("GPU cores not specified. Using CPU.")
        return torch.device("cpu"), []

    try:
        requested_gpu_ids = [
            int(value.strip())
            for value in gpu_cores.split(",")
            if value.strip() != ""
        ]
    except ValueError:
        raise ValueError(
            "-gpu-cores must be a comma-separated list "
            "of integer GPU IDs, for example: 0 or 0,1,2"
        )

    # Remove duplicates while preserving order.
    requested_gpu_ids = list(
        dict.fromkeys(requested_gpu_ids)
    )

    if len(requested_gpu_ids) == 0:
        raise ValueError(
            "-gpu-cores was provided but no GPU IDs were specified"
        )

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. "
            "Falling back to CPU."
        )
        return torch.device("cpu"), []

    available_gpu_count = torch.cuda.device_count()

    print(
        f"CUDA devices available: "
        f"{available_gpu_count}"
    )

    invalid_gpu_ids = [
        gpu_id
        for gpu_id in requested_gpu_ids
        if gpu_id < 0 or gpu_id >= available_gpu_count
    ]

    if invalid_gpu_ids:
        print(
            f"Requested GPU IDs not available: "
            f"{invalid_gpu_ids}. "
            f"Available GPU IDs are "
            f"0..{available_gpu_count - 1}. "
            f"Falling back to CPU."
        )
        return torch.device("cpu"), []

    for gpu_id in requested_gpu_ids:
        print(
            f"Using GPU {gpu_id}: "
            f"{torch.cuda.get_device_name(gpu_id)}"
        )

    return (
        torch.device(
            f"cuda:{requested_gpu_ids[0]}"
        ),
        requested_gpu_ids
    )

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
        "-confidence_threshold",
        "--confidence_threshold",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        type=float,
        help=(
            "Detection confidence threshold used by "
            "post_processing(). Default: 0.40"
        )
    )

    parser.add_argument(
        "-nms_iou_threshold",
        "--nms_iou_threshold",
        default=DEFAULT_NMS_IOU_THRESHOLD,
        type=float,
        help=(
            "NMS IoU threshold used by post_processing(). "
            "Default: 0.60"
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
        "-gpu-cores",
        "--gpu-cores",
        dest="gpu_cores",
        default=None,
        type=str,
        help=(
            "Optional comma-separated CUDA GPU IDs, "
            "for example: 0 or 0,1,2. "
            "If omitted, inference runs on CPU."
        )
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

    if not (
        0.0 <=
        args.confidence_threshold <=
        1.0
    ):
        raise ValueError(
            "confidence_threshold must satisfy "
            "0 <= threshold <= 1"
        )

    if not (
        0.0 <=
        args.nms_iou_threshold <=
        1.0
    ):
        raise ValueError(
            "nms_iou_threshold must satisfy "
            "0 <= threshold <= 1"
        )

    cfg = edict(dict(Cfg))
    cfg.dataset_dir = args.dataset_dir
    cfg.classes = args.classes

    device, gpu_ids = select_device(
        args.gpu_cores
    )

    (
        subset_label_path,
        test_image_paths,
        test_truth_lines,
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
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"NMS IoU threshold: {args.nms_iou_threshold}")
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

    # Use one image per forward pass so inference timing is
    # measured exactly per image on both CPU and GPU.
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=(
            4 if device.type == "cuda"
            else 0
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        collate_fn=val_collate
    )

    print("Evaluation batch size: 1")
    print(
        f"DataLoader workers: "
        f"{4 if device.type == 'cuda' else 0}"
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

    if device.type == "cuda" and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0]
        )

    model.eval()

    print(
        f"Inference device: {device}"
    )

    if len(gpu_ids) > 1:
        print(
            f"DataParallel GPUs: {gpu_ids}"
        )

    print("Running COCO evaluation...")

    try:
        (
            evaluator,
            average_inference_seconds
        ) = evaluate(
            model=model,
            data_loader=test_loader,
            cfg=cfg,
            device=device,
            image_paths=test_image_paths,
            truth_lines=test_truth_lines,
            confidence_threshold=(
                args.confidence_threshold
            ),
            nms_iou_threshold=(
                args.nms_iou_threshold
            ),
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
