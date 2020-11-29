FROM python:3.7

# prevents a bug with a libgl
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /app
COPY ./src /src

RUN wget https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt -O /face2vec_model.pth

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]