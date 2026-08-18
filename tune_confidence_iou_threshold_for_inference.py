# -*- coding: utf-8 -*-
#call: nohup python tune_confidence_iou_threshold_for_inference.py > test_tune.log 2>&1 &
# -confidence_threshold 0.1  -nms_iou_threshold 0.5 based on tune_confidence_iou_threshold_for_inference.py
import os
import re
import subprocess
import random
random.seed(40)

#params we want to try
CONFIDENCES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
IOUS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


#get 60% subset which will statistically lead us to same or similar optimal value without going through entire li
CONFIDENCES = sorted(
    random.sample(CONFIDENCES, int(len(CONFIDENCES) * 0.6))
)

IOUS = sorted(
    random.sample(IOUS, int(len(IOUS) * 0.6))
)
print("CONFIDENCES:", CONFIDENCES)
print("IOUS:", IOUS)
RESULTS_DIR = (
    "/content/gdrive/MyDrive/Smoke_and_Fire_Datasets/"
    "D-Fire/results/tuned"
)

COMMAND_BASE = [
    "python",
    "evaluate_on_test.py",
    "-checkpoint",
    "/content/gdrive/MyDrive/pytorch-YOLOv4/checkpoints/Yolov4_epoch125.pth",
    "-test_label",
    "/content/gdrive/MyDrive/pytorch-YOLOv4/data/val.txt", #must use validation set not test set
    "-dir",
    "/content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire",
    "-classes",
    "2",
    "-test_coverage_ratio",
    "0.1",
]


def parse_metrics(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    marker = "COCO metrics"
    index = text.rfind(marker)

    if index < 0:
        return None

    metrics_text = text[index:]

    def get_metric(name):
        match = re.search(
            rf"^{re.escape(name)}\s*:\s*([0-9.]+)",
            metrics_text,
            re.MULTILINE
        )
        return float(match.group(1)) if match else None

    ap = get_metric("AP")
    ar_small = get_metric("AR small")
    ar_med = get_metric("AR med")
    ar_large = get_metric("AR large")

    if None in (ap, ar_small, ar_med, ar_large):
        return None

    avg_ar = (
        ar_small +
        ar_med +
        ar_large
    ) / 3.0

    return ap, avg_ar


def print_progress(done, total):
    width = 30
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)

    print(
        f"\r[{bar}] {done}/{total}",
        end="",
        flush=True
    )


def main():
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    total = len(CONFIDENCES) * len(IOUS)
    done = 0

    best_ap = None
    best_ap_pair = None

    best_avg_ar = None
    best_avg_ar_pair = None

    for confidence in CONFIDENCES:
        for iou in IOUS:
            log_path = os.path.join(
                RESULTS_DIR,
                f"test_conf_{confidence:.1f}_iou_{iou:.1f}.log"
            )

            command = COMMAND_BASE + [
                "-confidence_threshold",
                str(confidence),
                "-nms_iou_threshold",
                str(iou),
            ]

            with open(
                log_path,
                "w",
                encoding="utf-8"
            ) as log_file:
                subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False
                )

            metrics = parse_metrics(
                log_path
            )

            if metrics is not None:
                ap, avg_ar = metrics

                if (
                    best_ap is None or
                    ap > best_ap
                ):
                    best_ap = ap
                    best_ap_pair = (
                        confidence,
                        iou
                    )

                if (
                    best_avg_ar is None or
                    avg_ar > best_avg_ar
                ):
                    best_avg_ar = avg_ar
                    best_avg_ar_pair = (
                        confidence,
                        iou
                    )

            done += 1
            print_progress(
                done,
                total
            )

    print()

    if best_ap_pair is not None:
        print(
            "Highest AP: "
            f"{best_ap:.3f} "
            f"at confidence={best_ap_pair[0]:.1f}, "
            f"iou={best_ap_pair[1]:.1f}"
        )

    if best_avg_ar_pair is not None:
        print(
            "Highest avg_AR: "
            f"{best_avg_ar:.3f} "
            f"at confidence={best_avg_ar_pair[0]:.1f}, "
            f"iou={best_avg_ar_pair[1]:.1f}"
        )


if __name__ == "__main__":
    main()
