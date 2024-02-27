FROM python:3.8-slim-buster

WORKDIR /app
EXPOSE 5000

RUN pip3 install opencv-python-headless

COPY . /app/

CMD ["python3", "./src/test.py"]