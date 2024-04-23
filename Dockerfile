FROM python:3.11-slim-buster

WORKDIR /app/
EXPOSE 5000

RUN pip3 install keras==3.2 tensorflow==2.16.1 Pillow Flask

COPY . /app/

CMD ["flask", "--app", "./src/http_test.py", "--debug", "run", "--host=0.0.0.0"]