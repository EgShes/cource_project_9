import os
import os.path as osp
import sqlite3 as sl
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import (MTCNN, InceptionResnetV1,
                             fixed_image_standardization)

from src.config import config
from src.image_utils import resize, resize_proportionally


class FaceDetInferencer:

    def __init__(self, max_input_size: int = 400, face_size: int = 160, min_prob: float = 0.9):
        self.face_size = face_size
        self.min_prob = min_prob
        self.net = MTCNN(keep_all=True).to(config.device).eval()

    @torch.no_grad()
    def run_network(self, image: np.ndarray) -> List[np.ndarray]:
        cropped_image, scale = resize_proportionally(image, self.face_size) 
        boxes, probs = self.net.detect(cropped_image)
        if boxes is None:
            return []
        boxes = boxes[probs >= self.min_prob]
        faces = self._crop_faces(image, boxes, scale)
        faces = [fixed_image_standardization(face) for face in faces]
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


class Face2VecInferencer:

    def __init__(self): 
        self.net = InceptionResnetV1(pretrained='vggface2').to(config.device).eval()

    @torch.no_grad()
    def run_network(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        faces = torch.FloatTensor(np.stack(faces)).permute(0, 3, 1, 2)
        face_vectors = self.net(faces)
        face_vectors = face_vectors.detach().cpu().numpy()
        face_vectors = [face_vectors[i,:] for i in range(face_vectors.shape[0])]
        return face_vectors


class DataBaseHandler:

    TABLE = 'MAIN'
    VEC_COL = 'vector'
    IDENT_COL = 'identifier'

    def __init__(self, db_path: str = config.db_path, db_name: str = config.db_name ):
        if not osp.isfile(osp.join(db_path, db_name)):
            self._init_db(db_path, db_name)
        self.path = osp.join(db_path, db_name)
        self._update_values()

    def _init_db(self, db_path: str, db_name: str):
        os.makedirs(db_path, exist_ok=True)
        connection = sl.connect(osp.join(db_path, db_name))
        with connection:
            connection.execute(
                f'create table {self.TABLE} (' \
                'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,' \
                f'{self.VEC_COL} BLOB NOT NULL,' \
                f'{self.IDENT_COL} TEXT NOT NULL UNIQUE );'
            )

    def _db2df(self) -> pd.DataFrame:
        connection = sl.connect(self.path)
        with connection:
            df = pd.read_sql('select * from main', connection)
            df[self.VEC_COL] = df[self.VEC_COL].apply(lambda x: np.frombuffer(x, dtype=np.float32))
            return df

    def _update_values(self):
        df = self._db2df()
        identifiers = df[self.IDENT_COL].to_list() if len(df) != 0 else None
        vectors = np.stack(df[self.VEC_COL].to_numpy()) if len(df) != 0 else None
        self.identifiers = identifiers
        self.vectors = vectors
    
    def add_record(self, vector: np.ndarray, identifier: str) -> bool:
        connection = sl.connect(self.path)
        with connection:
            df = pd.DataFrame({
                self.VEC_COL: [vector],
                self.IDENT_COL: [identifier]
            })
            try:
                df.to_sql(self.TABLE, connection, if_exists='append', index=False)
            except sl.IntegrityError as e:
                return False

        self._update_values()
        return True

    def delete_record(self, identifier: str) -> bool:
        connection = sl.connect(self.path)
        with connection:
            connection.execute(
                f'DELETE from {self.TABLE} where {self.IDENT_COL} = "{identifier}"'
            )
        self._update_values()

