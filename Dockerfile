FROM python:3.11-slim-buster

WORKDIR /app/
EXPOSE 5000

RUN pip3 install keras==3.2 tensorflow==2.16.1 matplotlib Pillow

COPY . /app/

CMD ["python3", "./src/test.py"]