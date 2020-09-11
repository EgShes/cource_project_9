from typing import List

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from .image_utils import resize, resize_proportionally

from .config import config


class FaceDetInferencer:

    def __init__(self, max_input_size: int = 400, face_size: int = 160, min_prob: float = 0.9):
        self.face_size = face_size
        self.min_prob = min_prob
        self.net = MTCNN(keep_all=True).to(config.device).eval()

    @torch.no_grad()
    def run_network(self, image: np.ndarray):
        cropped_image, scale = resize_proportionally(image, self.face_size) 
        boxes, probs = self.net.detect(cropped_image)
        if boxes is None:
            return []
        boxes = boxes[probs >= self.min_prob]
        faces = self._crop_faces(image, boxes, scale)
        return faces

    def _crop_faces(self, image: np.ndarray, boxes: np.ndarray, scale: float) -> List[np.ndarray]:
        faces = []
        boxes[boxes < 0] = 0
        boxes = boxes / scale
        boxes = boxes.astype(np.int)
        for i in range(boxes.shape[0]):
            x0, y0, x1, y1 = boxes[i,:]
            face = resize(image[y0:y1, x0:x1], self.face_size)
            faces.append(face)

        return faces
