%%bash
cd /content/gdrive/MyDrive/pytorch-YOLOv4/ || exit 1

PATH_BACKBONE_WEIGHTS="/content/gdrive/MyDrive/pytorch-YOLOv4/yolov4.conv.137.pth"
PATH_D_FIRE_DATA_ROOT="/content/gdrive/MyDrive/Smoke_and_Fire_Datasets/D-Fire"
PATH_TO_TRAIN_DATA_LIST="/content/gdrive/MyDrive/pytorch-YOLOv4/data/train.txt"

python3 train.py \
  -l 0.001 \
  -g 0 \
  -pretrained "$PATH_BACKBONE_WEIGHTS" \
  -classes 2 \
  -dir "$PATH_D_FIRE_DATA_ROOT" \
  -train_label_path "$PATH_TO_TRAIN_DATA_LIST"

