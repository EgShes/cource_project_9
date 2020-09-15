FROM python:3.7

# prevents a bug with a libgl
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /app
COPY ./src /src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]