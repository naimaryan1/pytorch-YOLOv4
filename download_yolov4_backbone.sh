if [ -f /content/gdrive/MyDrive/pytorch-YOLOv4/yolov4.conv.137.pth ]; then
    cp /content/gdrive/MyDrive/pytorch-YOLOv4/yolov4.conv.137.pth .
else
    gdown \
      --id 1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA \
      --output yolov4.conv.137.pth
fi

ls -lh yolov4.conv.137.pth data/train.txt data/val.txt
