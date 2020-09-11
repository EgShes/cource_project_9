import cv2
import numpy as np


def resize_proportionally(image: np.ndarray, size: int):
    max_cur_size = max(image.shape)
    scale = size / max_cur_size
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return image, scale


def resize(image: np.ndarray, size: int) -> np.ndarray:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return image
