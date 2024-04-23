from flask import Flask
from flask import request

from predict import evaluate

app = Flask(__name__)

@app.post("/sendimages")
def get_data():
    imageData = request.get_data().decode("utf-8")
    result = evaluate(imageData)
    return ("C" if result[0] < 0.5 else "D") + "\n"