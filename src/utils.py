import numpy as np


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    assert matrix.ndim == 2 and vector.ndim == 1
    assert matrix.shape[1] == vector.shape[0]
    sym = np.dot(matrix, vector) / (np.linalg.norm(matrix, axis=-1) * np.linalg.norm(vector))
    return sym