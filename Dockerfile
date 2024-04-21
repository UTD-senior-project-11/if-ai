FROM python:3.11-slim-buster

EXPOSE 5000

WORKDIR /app/

COPY ./src/ ./src/
COPY ./build/requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["python3", "./src/dataset_preprocessing.py"]
CMD ["python3", "./src/main.py"]
