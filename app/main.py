import uuid

import imageio
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from pydantic import conlist

from app.models import AddFaceResponse
from src import DataBaseHandler, Face2VecInferencer, FaceDetInferencer, config

db_handler = DataBaseHandler(db_path=config.db_path, db_name=config.db_name)

facedet_infer = FaceDetInferencer(
    max_input_size=config.max_input_size,
    face_size=config.face_size,
    min_prob=config.man_face_prob
)
face2vec_infer = Face2VecInferencer()

app = FastAPI()

@app.post(
    '/add_face',
    summary='Добавить лицо в базу данных',
    response_model=AddFaceResponse,
)
def add_face(image: bytes = File(...)):
    """
    На вход принимает одно изображение. 

    Если на фото нет лиц или лиц больше, чем одно, вернется ошибка 404.
    
    В случае успеха венется ответ с кодом 200 и уникальной строкой (uuid), присвоенной данному лицу.
    """
    image = imageio.imread(image)
    vector = image2vector(image)
    identifier = str(uuid.uuid4())
    success = db_handler.add_record(vector, identifier)
    if success:
        return {'uuid': identifier, 'message': 'face successfully added'}
    else:
        return {'message': 'face was not added'}


def image2vector(image: imageio.core.util.Array):
    image = np.array(image)
    faces = facedet_infer.run_network(image)
    print(len(faces))
    if len(faces) == 0:
        raise HTTPException(422, 'no face detected')
    elif len(faces) > 1:
        raise HTTPException(422, 'more than one face detected')
    else:
        vector = face2vec_infer.run_network(faces)[0]
        return vector
