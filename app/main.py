import uuid

import imageio
import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from pydantic import conlist

from app.models import AddFaceResponse, DeleteFaceResponse, DetectFaceResponse
from src import DataBaseHandler, Face2VecInferencer, FaceDetInferencer, config, cosine_similarity

db_handler = DataBaseHandler(db_path=config.db_path, db_name=config.db_name)

facedet_infer = FaceDetInferencer(
    max_input_size=config.max_input_size,
    face_size=config.face_size,
    min_prob=config.man_face_prob
)
face2vec_infer = Face2VecInferencer()

app = FastAPI(
    title="Вахтер тетя Галя",
    description="Знает, кто живет в подъезде и как он выглядит",
    version="0.1.0",
)


@app.post(
    '/detect_face',
    summary='Найти лицо на изображении',
    response_model=DetectFaceResponse,
    tags=['Для промышленного использования']
)
def detect_face(image: bytes = File(...)):
    """
    На вход принимает одно изображение. 

    Если на фото нет лиц или лиц больше, чем одно, вернется ошибка 404.
    
    В случае, если в базу данных есть похожее на это лицо, вернется ответ с кодом 200 
    и уникальной строкой, соответсвующей наиболее похожему лицу. Иначе - 404.
    """
    if db_handler.vectors is None:
        return {'message': 'database has no faces'}
    image = imageio.imread(image)
    vector = image2vector(image)
    cosine = cosine_similarity(db_handler.vectors, vector)
    if cosine[cosine.argmax()] >= config.similarity_threshold:
        identifier = db_handler.identifiers[cosine.argmax()]
        return {'uuid': identifier, 'message': 'face was identified'}
    else:
        return {'message': 'face was not identified'}


@app.post(
    '/add_face',
    summary='Добавить лицо в базу данных',
    response_model=AddFaceResponse,
    tags=['Для промышленного использования']
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


@app.delete(
    '/delete_face',
    summary='Удалить лицо из базы данных',
    response_model=DeleteFaceResponse,
    tags=['Для промышленного использования']
)
def delete_face(uuid: uuid.UUID):
    """
    На вход принимает uuid - уникальную для лица строку, которая была получена при добавлении лица.
    
    Всегда возвращает 200, даже если такого лица не было.
    """
    db_handler.delete_record(str(uuid))
    return {'message': 'face was deleted'}


@app.get(
    '/see_uuids',
    summary='Получить все uuid с сервера',
    tags=['Для разработки']
)
def see_uuids():
    """
    Данный метод предназначен для удобства отладки.
    """
    return {'uuids': db_handler.identifiers}


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
