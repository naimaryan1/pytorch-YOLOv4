# -*- coding: utf-8 -*-

import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

# Use the corrected 2-class Darknet CFG
Cfg.use_darknet_cfg = False
Cfg.cfgfile = os.path.join(
    _BASE_DIR,
    "cfg",
    "yolov4-custom-2class-t4.cfg"
)

# T4-friendly settings
Cfg.batch = 32
Cfg.subdivisions = 8

Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3

Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = 0.1

Cfg.learning_rate = 0.001
Cfg.burn_in = 1000
Cfg.max_batches = 4000
Cfg.steps = [3200, 3600]
Cfg.policy = Cfg.steps
Cfg.scales = (0.1, 0.1)

Cfg.cutmix = 0
Cfg.mosaic = 0
Cfg.mixup = 0

Cfg.letter_box = 0
Cfg.jitter = 0.2

# Smoke and fire
Cfg.classes = 2

Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height

Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60

Cfg.TRAIN_EPOCHS = 50

# This repository expects converted annotation text files
Cfg.train_label = os.path.join(
    _BASE_DIR,
    "data",
    "train.txt"
)

Cfg.val_label = os.path.join(
    _BASE_DIR,
    "data",
    "val.txt"
)

Cfg.TRAIN_OPTIMIZER = "adam"

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3
else:
    Cfg.mixup = 0

Cfg.checkpoints = os.path.join(
    _BASE_DIR,
    "checkpoints"
)

Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(
    _BASE_DIR,
    "log"
)

Cfg.iou_type = "iou"
Cfg.keep_checkpoint_max = 10