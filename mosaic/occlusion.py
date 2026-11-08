"""occlusion.py - what's in front of the face: a hand, hair, a cup, a microphone. The mosaic isn't drawn
there, so they stay in front of it instead of being painted over.

The model is FaceFusion's occluder (XSeg, models/xseg_1.onnx, downloaded by run.bat): it looks at the face
lined up the standard way (the five points of InsightFace's ArcFace layout: both eyes, the nose, the mouth's
corners, at 256 x 256) and says, for every pixel, face or not.

On an NVIDIA GPU (onnxruntime-gpu) it takes about 30 ms, so it runs every other frame and the mask in
between follows the face; on the CPU it's about 130 ms, so it runs every few frames. Without the model,
or with occlusion off, the mosaic covers the whole face as before.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "xseg_1.onnx")
SIZE = 256

# InsightFace's ArcFace layout (112 x 112), moved to 256 x 256 the way InsightFace does for 128 / 256
_ARCFACE = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]])
TEMPLATE = _ARCFACE * (SIZE / 128) + [8 * SIZE / 128, 0]

# The same five points on MediaPipe's face mesh: the eye on the picture's left (the person's right) first
EYE_A, EYE_B, NOSE, MOUTH_A, MOUTH_B = (33, 133), (362, 263), 1, 61, 291


def five_points(lm):
    return np.float32([lm[list(EYE_A)].mean(0), lm[list(EYE_B)].mean(0), lm[NOSE], lm[MOUTH_A], lm[MOUTH_B]])


class Occluder:
    def __init__(self):
        self.session = None
        self.gpu = False
        self.on = False
        self.why = ""
        self._mask = None          # the last mask, in the lined-up face's frame
        self._frames = 0
        if not os.path.exists(MODEL):
            self.why = "no models/xseg_1.onnx (run.bat downloads it)"
            return
        try:
            import torch  # noqa: F401  its CUDA DLLs, for onnxruntime-gpu
        except Exception:
            pass
        try:
            import onnxruntime as ort
        except Exception as e:
            self.why = f"no onnxruntime ({e})"
            return
        providers = [p for p in (("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                                 "CPUExecutionProvider")
                     if (p[0] if isinstance(p, tuple) else p) in ort.get_available_providers()]
        self.session = ort.InferenceSession(MODEL, providers=providers)
        self.gpu = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self.input = self.session.get_inputs()[0].name
        self.every = 2 if self.gpu else 4          # how often it really looks: the mask follows the face between
        self.on = True

    def mask(self, frame, lm, box):
        """A float mask over box (x0, y0, x1, y1) of frame: 1 where the face shows, 0 where something's in front."""
        x0, y0, x1, y1 = box
        m, _ = cv2.estimateAffinePartial2D(five_points(lm), TEMPLATE)
        if m is None:
            return np.ones((y1 - y0, x1 - x0), np.float32)
        if self._mask is None or self._frames % self.every == 0:
            crop = cv2.warpAffine(frame, m, (SIZE, SIZE), borderValue=0.0)
            out = self.session.run(None, {self.input: crop[None].astype(np.float32) / 255.0})[0][0, ..., 0]
            out = cv2.GaussianBlur(np.clip(out, 0, 1), (0, 0), 5)
            self._mask = ((np.clip(out, 0.5, 1) - 0.5) * 2).astype(np.float32)   # firm, softened (as FaceFusion)
        self._frames += 1
        back = cv2.invertAffineTransform(m)
        back[:, 2] -= (x0, y0)
        # outside the lined-up crop (the top of the forehead, the hair line) counts as face
        return cv2.warpAffine(self._mask, back, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
