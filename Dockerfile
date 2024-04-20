FROM python:3.11-slim-buster

WORKDIR /app/
EXPOSE 5000

COPY . /app/

RUN pip3 install -r build/requirements.txt

CMD ["python3", "./src/main.py"]
