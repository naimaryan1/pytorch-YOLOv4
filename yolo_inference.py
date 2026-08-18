"""
sudo /home/edgemonitor/client_venvs/yolo_inference/bin/python \

python yolo_inference_multi_camera.py \
  --data-input-path \
    /dev/shm/camera/0/ \
    /dev/shm/camera/1/  \
  --model-path /mnt/edgeusb/client_models/model.onnx \
  --model-type onnx \
  --confidence-threshold 0.30 \
  --nms-iou-threshold 0.50

  example:

sudo /home/edgemonitor/client_venvs/yolo_inference/bin/python yolo_inference.py \
  --data-input-path \
    /dev/shm/camera/0/latest_frame \
    /dev/shm/camera/1/latest_frame \
  --model-path /mnt/edgeusb/client_models/yolov4_1_3_608_608_static.onnx \
  --model-type onnx \
  --confidence-threshold 0.30 \
  --nms-iou-threshold 0.50

"""
import argparse
import importlib.util
import json
import mmap
import os
import pickle
import re
import struct
import time
import urllib.request
from enum import Enum

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


SHM_PATH = "/dev/shm/edgemonitor_camera_0_latest"
MODEL_DIR = "/mnt/edgeusb/client_models"
DEFAULT_OUTPUT_DIR = "/mnt/edgeusb/client_model_outputs"
DEFAULT_DB_PATH = "/mnt/edgeusb/edgemonitor/rocksdb"
HEADER = struct.Struct("<QqII8sQ")
PTS_CLOCK_HZ = 90000
NS_PER_SECOND = 1_000_000_000
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
DEFAULT_NMS_IOU_THRESHOLD = 0.50


class DATA_INPUT_TYPE(Enum):
    LOCAL_FILE_PATH = 1
    LOCAL_DB_CONNECTION = 2
    URL = 3


class DATA_OUTPUT_TYPE(Enum):
    LOCAL_FILE_PATH = 1
    LOCAL_DB_CONNECTION = 2
    URL = 3


class MODEL_TYPE(Enum):
    sklearn = 1
    xgboost = 2
    pytorch = 3
    tensorflow = 4
    onnx = 5
    custom_python = 6


class ResultWriter:
    def __init__(self, data_output_path, output_type):
        self.data_output_path = data_output_path
        self.output_type = output_type
        self.connection = None

    def __enter__(self):
        if self.output_type == DATA_OUTPUT_TYPE.LOCAL_DB_CONNECTION:
            self.connection = _open_key_value_db(self.data_output_path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection is not None:
            self.connection.close()

    def requires_frame_image(self):
        return (
            self.output_type == DATA_OUTPUT_TYPE.LOCAL_FILE_PATH
            and _is_directory_output_path(self.data_output_path)
        )

    def write(self, result, bgr=None):
        if self.output_type == DATA_OUTPUT_TYPE.LOCAL_FILE_PATH:
            _write_local_file_output(self.data_output_path, result, bgr)
        elif self.output_type == DATA_OUTPUT_TYPE.LOCAL_DB_CONNECTION:
            self._write_key_value(result)
        elif self.output_type == DATA_OUTPUT_TYPE.URL:
            _post_json(self.data_output_path, result)
        else:
            raise ValueError(f"Unsupported output type: {self.output_type}")

    def _write_key_value(self, result):
        if self.connection is None:
            raise RuntimeError("Key/value DB output is not open")

        key = _result_key(result)
        value = json.dumps(result).encode("utf-8")
        self.connection.put(key.encode("utf-8"), value)


def my_inference(
    data_input_path: str,
    input_type: DATA_INPUT_TYPE,
    data_output_path: str,
    output_type: DATA_OUTPUT_TYPE,
    model_path: str,
    model_type: MODEL_TYPE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
) -> None:
    _run_inference_loop(
        data_input_path,
        _coerce_enum(DATA_INPUT_TYPE, input_type),
        data_output_path,
        _coerce_enum(DATA_OUTPUT_TYPE, output_type),
        model_path,
        _coerce_enum(MODEL_TYPE, model_type),
        max_frames=None,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )


def _run_inference_loop(
    data_input_path,
    input_type,
    data_output_path,
    output_type,
    model_path,
    model_type,
    max_frames,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    nms_iou_threshold=DEFAULT_NMS_IOU_THRESHOLD,
):
    if input_type != DATA_INPUT_TYPE.LOCAL_FILE_PATH:
        raise NotImplementedError(
            "This proof of concept reads I420 frames from local "
            "shared-memory files only."
        )

    if isinstance(data_input_path, str):
        data_input_paths = [data_input_path]
    else:
        data_input_paths = list(data_input_path)

    if len(data_input_paths) == 0:
        raise ValueError("At least one --data-input-path is required")

    resolved_model_path = _resolve_model_path(model_path)
    model = _load_model(resolved_model_path, model_type)
    frames_written = 0

    shm_files = []
    camera_streams = []

    try:
        for path in data_input_paths:
            shm_file = open(path, "rb", buffering=0)
            memory = mmap.mmap(
                shm_file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )

            shm_files.append(shm_file)
            camera_streams.append(
                {
                    "path": path,
                    "memory": memory,
                    "camera_id": _camera_id_from_path(path),
                    "stream_start_ns": None,
                    "last_sequence": None,
                }
            )

        with ResultWriter(data_output_path, output_type) as writer:
            while True:
                processed_frame = False

                for stream in camera_streams:
                    frame = _read_latest_i420_frame(
                        stream["memory"]
                    )

                    if frame is None:
                        continue

                    if frame["sequence"] == stream["last_sequence"]:
                        continue

                    if stream["stream_start_ns"] is None:
                        stream["stream_start_ns"] = frame["timestamp_ns"]

                    video_pts_90khz = _video_pts_90khz(
                        frame["timestamp_ns"],
                        stream["stream_start_ns"],
                    )

                    detections = []
                    bgr = None

                    if model is not None or writer.requires_frame_image():
                        bgr = _i420_to_bgr(
                            frame["i420"],
                            frame["width"],
                            frame["height"],
                        )

                    if model is not None:
                        detections = _run_detections(
                            model,
                            model_type,
                            bgr,
                            confidence_threshold,
                            nms_iou_threshold,
                        )

                    result = {
                        "camera_id": stream["camera_id"],
                        "frame_sequence": frame["sequence"],
                        "capture_timestamp_ns": frame["timestamp_ns"],
                        "video_pts_90khz": video_pts_90khz,
                        "detections": detections,
                    }

                    writer.write(result, bgr)

                    frames_written += 1
                    stream["last_sequence"] = frame["sequence"]
                    processed_frame = True

                    if (
                        max_frames is not None
                        and frames_written >= max_frames
                    ):
                        return

                if not processed_frame:
                    time.sleep(0.001)

    finally:
        for stream in camera_streams:
            stream["memory"].close()

        for shm_file in shm_files:
            shm_file.close()


def _read_latest_i420_frame(memory):
    if len(memory) < HEADER.size:
        return None

    (
        sequence_start,
        timestamp_ns,
        width,
        height,
        pixel_format,
        byte_count,
    ) = HEADER.unpack_from(memory, 0)

    if sequence_start == 0 or pixel_format.rstrip(b"\0") != b"I420":
        return None

    expected_byte_count = width * height * 3 // 2
    frame_start = HEADER.size
    frame_end = frame_start + byte_count

    if (
        width == 0
        or height == 0
        or byte_count != expected_byte_count
        or frame_end > len(memory)
    ):
        return None

    frame_bytes = memory[frame_start:frame_end]
    sequence_end = struct.unpack_from("<Q", memory, 0)[0]
    if sequence_start != sequence_end:
        return None

    return {
        "sequence": int(sequence_start),
        "timestamp_ns": int(timestamp_ns),
        "width": int(width),
        "height": int(height),
        "i420": frame_bytes,
    }


def _load_model(model_path, model_type):
    if not model_path:
        return None

    if model_type == MODEL_TYPE.custom_python:
        return _load_custom_python_model(model_path)
    if model_type == MODEL_TYPE.pytorch:
        import torch

        model = torch.load(model_path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
        return model
    if model_type == MODEL_TYPE.onnx:
        import onnxruntime

        return onnxruntime.InferenceSession(model_path)
    if model_type == MODEL_TYPE.tensorflow:
        import tensorflow as tf

        return tf.keras.models.load_model(model_path)
    if model_type == MODEL_TYPE.sklearn:
        try:
            import joblib

            return joblib.load(model_path)
        except ImportError:
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
    if model_type == MODEL_TYPE.xgboost:
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(model_path)
        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def _load_custom_python_model(model_path):
    spec = importlib.util.spec_from_file_location(
        "client_custom_model",
        model_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load custom Python model: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_detections(
    model,
    model_type,
    bgr,
    confidence_threshold,
    nms_iou_threshold,
):
    if model is None:
        return []

    if model_type == MODEL_TYPE.custom_python:
        return _run_custom_python_detections(model, bgr)
    if model_type == MODEL_TYPE.pytorch:
        return _run_pytorch_detections(model, bgr)
    if model_type == MODEL_TYPE.onnx:
        return _run_onnx_detections(
            model,
            bgr,
            confidence_threshold,
            nms_iou_threshold,
        )

    predict = getattr(model, "predict", None)
    if callable(predict):
        return _normalize_detections(predict([bgr]))

    return []


def _run_custom_python_detections(module, bgr):
    for function_name in ("detect", "infer", "run_inference"):
        detector = getattr(module, function_name, None)
        if callable(detector):
            return _normalize_detections(detector(bgr))
    return []


def _run_pytorch_detections(model, bgr):
    if not callable(model):
        return []

    import torch

    rgb = bgr[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0) / 255.0

    with torch.no_grad():
        output = model(tensor)

    return _normalize_detections(output)



def _run_onnx_detections(
    session,
    bgr,
    confidence_threshold,
    nms_iou_threshold,
):
    numpy = _require_numpy()

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    input_height = 608
    input_width = 608

    if (
        len(input_shape) == 4
        and isinstance(input_shape[2], int)
        and isinstance(input_shape[3], int)
    ):
        input_height = input_shape[2]
        input_width = input_shape[3]

    resized = cv2.resize(
        bgr,
        (input_width, input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(numpy.float32) / 255.0
    tensor = numpy.transpose(tensor, (2, 0, 1))
    tensor = numpy.expand_dims(tensor, axis=0)

    outputs = session.run(
        None,
        {input_name: tensor},
    )

    if len(outputs) < 2:
        raise RuntimeError(
            "YOLOv4 ONNX model must return box and confidence outputs"
        )

    box_array = numpy.asarray(outputs[0])
    confs = numpy.asarray(outputs[1])

    if box_array.ndim == 4:
        box_array = box_array[:, :, 0, :]

    if box_array.ndim != 3 or box_array.shape[-1] != 4:
        raise RuntimeError(
            f"Unexpected YOLOv4 ONNX box output shape: {box_array.shape}"
        )

    if confs.ndim != 3:
        raise RuntimeError(
            f"Unexpected YOLOv4 ONNX confidence output shape: {confs.shape}"
        )

    boxes = box_array[0]
    class_scores = confs[0]

    max_conf = numpy.max(class_scores, axis=1)
    class_ids = numpy.argmax(class_scores, axis=1)

    keep_conf = max_conf > confidence_threshold
    boxes = boxes[keep_conf]
    max_conf = max_conf[keep_conf]
    class_ids = class_ids[keep_conf]

    frame_height, frame_width = bgr.shape[:2]
    detections = []

    for class_id in range(class_scores.shape[1]):
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_confs = max_conf[class_mask]

        if class_boxes.shape[0] == 0:
            continue

        keep = _nms_cpu(
            class_boxes,
            class_confs,
            nms_iou_threshold,
        )

        for index in keep:
            x1 = float(class_boxes[index, 0]) * frame_width
            y1 = float(class_boxes[index, 1]) * frame_height
            x2 = float(class_boxes[index, 2]) * frame_width
            y2 = float(class_boxes[index, 3]) * frame_height

            detections.append(
                {
                    "xyxy": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                    "class": int(class_id),
                    "confidence": float(class_confs[index]),
                }
            )

    return detections


def _nms_cpu(boxes, confs, nms_thresh=0.5):
    numpy = _require_numpy()

    if len(boxes) == 0:
        return numpy.array([], dtype=numpy.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = numpy.maximum(0.0, x2 - x1) * numpy.maximum(0.0, y2 - y1)
    order = confs.argsort()[::-1]

    keep = []

    while order.size > 0:
        current = order[0]
        keep.append(current)

        others = order[1:]
        if others.size == 0:
            break

        xx1 = numpy.maximum(x1[current], x1[others])
        yy1 = numpy.maximum(y1[current], y1[others])
        xx2 = numpy.minimum(x2[current], x2[others])
        yy2 = numpy.minimum(y2[current], y2[others])

        width = numpy.maximum(0.0, xx2 - xx1)
        height = numpy.maximum(0.0, yy2 - yy1)
        intersection = width * height

        union = areas[current] + areas[others] - intersection
        iou = numpy.divide(
            intersection,
            union,
            out=numpy.zeros_like(intersection),
            where=union > 0,
        )

        remaining = numpy.where(iou <= nms_thresh)[0]
        order = order[remaining + 1]

    return numpy.asarray(keep, dtype=numpy.int64)


def _i420_to_bgr(frame_bytes, width, height):
    numpy = _require_numpy()
    i420 = numpy.frombuffer(frame_bytes, dtype=numpy.uint8).reshape(
        (height * 3 // 2, width)
    )

    if cv2 is not None:
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)

    frame_bytes = i420.reshape(-1)
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    y_plane = frame_bytes[:y_size].reshape(height, width)
    u_plane = frame_bytes[y_size:y_size + uv_size].reshape(
        height // 2,
        width // 2,
    )
    v_plane = frame_bytes[y_size + uv_size:y_size + uv_size * 2].reshape(
        height // 2,
        width // 2,
    )

    y = y_plane.astype(numpy.float32)
    u = _upsample_chroma(u_plane, height, width).astype(numpy.float32) - 128.0
    v = _upsample_chroma(v_plane, height, width).astype(numpy.float32) - 128.0

    b = y + 1.772 * u
    g = y - 0.344136 * u - 0.714136 * v
    r = y + 1.402 * v
    return numpy.clip(numpy.dstack((b, g, r)), 0, 255).astype(numpy.uint8)


def _upsample_chroma(plane, height, width):
    return np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)[:height, :width]


def _annotate_frame(bgr, detections):
    if bgr is None:
        raise ValueError("A BGR frame is required for annotated frame output")

    annotated = bgr.copy()
    if cv2 is None:
        return annotated

    for detection in detections:
        box = _box_from_detection(detection)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = _detection_label(detection)
        if label:
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return annotated


def _box_from_detection(detection):
    if not isinstance(detection, dict):
        return None

    if "xyxy" in detection:
        return _coerce_box(detection["xyxy"], "xyxy")
    if "bbox_xyxy" in detection:
        return _coerce_box(detection["bbox_xyxy"], "xyxy")
    if "xywh" in detection:
        return _coerce_box(detection["xywh"], "xywh")
    if "bbox_xywh" in detection:
        return _coerce_box(detection["bbox_xywh"], "xywh")
    if "bbox" in detection:
        box_format = detection.get("bbox_format", "xyxy")
        return _coerce_box(detection["bbox"], box_format)
    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        return _coerce_box(
            [
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            ],
            "xyxy",
        )
    if all(key in detection for key in ("x", "y", "width", "height")):
        return _coerce_box(
            [
                detection["x"],
                detection["y"],
                detection["width"],
                detection["height"],
            ],
            "xywh",
        )

    return None


def _coerce_box(values, box_format):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None

    x1, y1, x2, y2 = [int(round(float(value))) for value in values]
    if box_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2

    return x1, y1, x2, y2


def _detection_label(detection):
    label = detection.get("label", detection.get("class", ""))
    score = detection.get("score", detection.get("confidence"))
    if score is None:
        return str(label) if label else ""

    return f"{label} {float(score):.2f}".strip()


def _normalize_detections(raw_detections):
    raw_detections = _to_jsonable(raw_detections)

    if raw_detections is None:
        return []
    if isinstance(raw_detections, dict):
        return [raw_detections]
    if isinstance(raw_detections, list):
        return raw_detections

    return [{"value": raw_detections}]


def _to_jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_local_file_output(data_output_path, result, bgr):
    if data_output_path == "-":
        _write_json_line(data_output_path, result)
        return

    if not _is_directory_output_path(data_output_path):
        _write_json_line(data_output_path, result)
        return

    os.makedirs(data_output_path, exist_ok=True)
    base_path = _result_base_path(data_output_path, result)
    image_path = base_path + ".jpg"
    json_path = base_path + ".json"
    annotated = _annotate_frame(bgr, result["detections"])

    if cv2 is None:
        raise ImportError("OpenCV is required to write annotated frame files")
    if not cv2.imwrite(image_path, annotated):
        raise RuntimeError(f"Failed to write annotated frame: {image_path}")

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


def _write_json_line(data_output_path, result):
    line = json.dumps(result)
    if data_output_path == "-":
        print(line, flush=True)
        return

    parent_dir = os.path.dirname(data_output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(data_output_path, "a", encoding="utf-8") as output_file:
        output_file.write(line + "\n")


def _is_directory_output_path(data_output_path):
    if data_output_path == "-":
        return False
    if data_output_path.endswith(os.sep):
        return True
    if os.path.isdir(data_output_path):
        return True
    return os.path.splitext(os.path.basename(data_output_path))[1] == ""


def _result_base_path(output_dir, result):
    filename = "camera_{camera_id}_frame_{frame_sequence}".format(**result)
    return os.path.join(output_dir, filename)


def _post_json(url, result):
    request = urllib.request.Request(
        url,
        data=json.dumps(result).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2):
        return


class KeyValueDb:
    def __init__(self, backend, db):
        self.backend = backend
        self.db = db

    def put(self, key, value):
        if self.backend == "rocksdb":
            self.db.put(key, value)
            return
        if self.backend == "rocksdict":
            self.db[key] = value
            self.db.flush()
            return

        raise RuntimeError(f"Unsupported key/value DB backend: {self.backend}")

    def close(self):
        close = getattr(self.db, "close", None)
        if callable(close):
            close()


def _open_key_value_db(data_output_path):
    db_path = _db_path(data_output_path)
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        import rocksdb

        options = rocksdb.Options(create_if_missing=True)
        return KeyValueDb("rocksdb", rocksdb.DB(db_path, options))
    except ImportError:
        pass

    try:
        from rocksdict import Options, Rdict

        options = Options()
        options.create_if_missing(True)
        return KeyValueDb("rocksdict", Rdict(db_path, options))
    except ImportError:
        pass

    raise ImportError(
        "LOCAL_DB_CONNECTION requires a RocksDB Python binding such as "
        "'rocksdb' or 'rocksdict'."
    )


def _db_path(data_output_path):
    if not data_output_path or data_output_path == DEFAULT_OUTPUT_DIR:
        return DEFAULT_DB_PATH
    if isinstance(data_output_path, dict):
        return data_output_path.get("db_path", DEFAULT_DB_PATH)
    if data_output_path.startswith("rocksdb://"):
        return data_output_path.removeprefix("rocksdb://")
    if data_output_path.lstrip().startswith("{"):
        config = json.loads(data_output_path)
        return config.get("db_path", DEFAULT_DB_PATH)
    return data_output_path


def _result_key(result):
    return "camera:{camera_id}:frame:{frame_sequence}".format(**result)


def _resolve_model_path(model_path):
    if not model_path:
        return None
    if os.path.isabs(model_path):
        return model_path
    return os.path.join(MODEL_DIR, model_path)


def _camera_id_from_path(data_input_path):
    match = re.search(r"camera_(\d+)_latest", data_input_path)
    if match:
        return int(match.group(1))
    return 0


def _video_pts_90khz(capture_ns, stream_start_ns):
    return (capture_ns - stream_start_ns) * PTS_CLOCK_HZ // NS_PER_SECOND


def _require_numpy():
    if np is None:
        raise ImportError("NumPy is required to convert I420 frames to BGR")
    return np


def _coerce_enum(enum_type, value):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in enum_type.__members__:
            return enum_type[normalized]
        for enum_value in enum_type:
            if enum_value.name.lower() == normalized.lower():
                return enum_value
    return enum_type(value)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-input-path",
        nargs="+",
        default=[SHM_PATH],
    )
    parser.add_argument(
        "--input-type",
        default=DATA_INPUT_TYPE.LOCAL_FILE_PATH.name,
        choices=[item.name for item in DATA_INPUT_TYPE],
    )
    parser.add_argument("--data-output-path", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-type",
        default=DATA_OUTPUT_TYPE.LOCAL_FILE_PATH.name,
        choices=[item.name for item in DATA_OUTPUT_TYPE],
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/edgeusb/client_models/Yolov4_epoch125.pth",
    )
    parser.add_argument(
        "--model-type",
        default=MODEL_TYPE.pytorch.name,
        choices=[item.name for item in MODEL_TYPE],
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=DEFAULT_NMS_IOU_THRESHOLD,
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _run_inference_loop(
        args.data_input_path,
        _coerce_enum(DATA_INPUT_TYPE, args.input_type),
        args.data_output_path,
        _coerce_enum(DATA_OUTPUT_TYPE, args.output_type),
        args.model_path,
        _coerce_enum(MODEL_TYPE, args.model_type),
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )


if __name__ == "__main__":
    main()
