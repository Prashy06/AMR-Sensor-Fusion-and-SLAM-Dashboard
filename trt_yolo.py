import os
import sys
import ctypes
import time
import cv2
import numpy as np
import tensorrt as trt

# COCO 80 Class Names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class TensorRTYOLO:
    # Resolve default engine path relative to this file's directory — portable across any clone
    _DEFAULT_ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.engine")

    def __init__(self, engine_path=None, conf_thresh=0.35, nms_thresh=0.45):
        self.engine_path = engine_path if engine_path is not None else self._DEFAULT_ENGINE
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # Load CUDA Runtime via ctypes with strict 64-bit argtypes
        self.cudart = ctypes.CDLL('libcudart.so')
        self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.cudart.cudaMalloc.restype = ctypes.c_int
        self.cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self.cudart.cudaFree.restype = ctypes.c_int
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.cudart.cudaMemcpy.restype = ctypes.c_int
        self.cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.cudart.cudaStreamCreate.restype = ctypes.c_int
        self.cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cudart.cudaStreamSynchronize.restype = ctypes.c_int
        
        # Load TensorRT Engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")
            
        with open(engine_path, 'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate device memory buffers
        self.d_input = ctypes.c_void_p()
        self.d_output = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        
        self.input_bytes = 1 * 3 * 640 * 640 * 4   # float32
        self.output_bytes = 1 * 84 * 8400 * 4     # float32
        
        self.cudart.cudaMalloc(ctypes.byref(self.d_input), self.input_bytes)
        self.cudart.cudaMalloc(ctypes.byref(self.d_output), self.output_bytes)
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))
        
        self.context.set_tensor_address('images', self.d_input.value)
        self.context.set_tensor_address('output0', self.d_output.value)
        
        # Pre-allocate pinned host output buffer
        self.h_output = np.empty((1, 84, 8400), dtype=np.float32)
        
        # Warmup GPU
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        self.infer_raw(dummy)
        print(f"[TensorRTYOLO] GPU TensorRT engine loaded and warmed up successfully: {engine_path}", flush=True)

    def infer_raw(self, input_tensor):
        # Host to Device (cudaMemcpyHostToDevice = 1)
        self.cudart.cudaMemcpy(self.d_input, input_tensor.ctypes.data, self.input_bytes, 1)
        self.context.execute_async_v3(self.stream.value)
        # Device to Host (cudaMemcpyDeviceToHost = 2)
        self.cudart.cudaMemcpy(self.h_output.ctypes.data, self.d_output, self.output_bytes, 2)
        self.cudart.cudaStreamSynchronize(self.stream.value)
        return self.h_output

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        input_tensor = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.ascontiguousarray(input_tensor.transpose(2, 0, 1)[np.newaxis, ...])
        return input_tensor, r, dw, dh

    def detect(self, frame):
        t0 = time.perf_counter()
        orig_h, orig_w = frame.shape[:2]
        input_tensor, ratio, dw, dh = self.preprocess(frame)
        
        raw_out = self.infer_raw(input_tensor) # (1, 84, 8400)
        out = raw_out[0].T # (8400, 84)

        boxes = out[:, :4]       # cx, cy, w, h
        scores = out[:, 4:]      # 80 class probs
        
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        mask = max_scores >= self.conf_thresh
        valid_boxes = boxes[mask]
        valid_scores = max_scores[mask]
        valid_classes = class_ids[mask]

        detections = []
        if len(valid_boxes) > 0:
            cx = (valid_boxes[:, 0] - dw) / ratio
            cy = (valid_boxes[:, 1] - dh) / ratio
            w = valid_boxes[:, 2] / ratio
            h = valid_boxes[:, 3] / ratio
            
            x1 = np.clip(cx - w / 2, 0, orig_w)
            y1 = np.clip(cy - h / 2, 0, orig_h)
            x2 = np.clip(cx + w / 2, 0, orig_w)
            y2 = np.clip(cy + h / 2, 0, orig_h)

            cv_boxes = [[int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])] for i in range(len(x1))]
            indices = cv2.dnn.NMSBoxes(cv_boxes, valid_scores.tolist(), self.conf_thresh, self.nms_thresh)

            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for idx in indices:
                    cid = int(valid_classes[idx])
                    name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"cls_{cid}"
                    detections.append({
                        'box': [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
                        'score': float(valid_scores[idx]),
                        'class_id': cid,
                        'class_name': name
                    })

        infer_ms = (time.perf_counter() - t0) * 1000.0
        return detections, infer_ms

    def __del__(self):
        try:
            if hasattr(self, 'd_input') and self.d_input.value:
                self.cudart.cudaFree(self.d_input)
            if hasattr(self, 'd_output') and self.d_output.value:
                self.cudart.cudaFree(self.d_output)
        except Exception:
            pass
