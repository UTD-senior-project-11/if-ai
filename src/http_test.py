from flask import Flask
from flask import request

from predict import evaluate

app = Flask(__name__)

@app.post("/sendimages")
def get_data():
    imageData = request.json.get('imageData')
    result = evaluate(imageData)
    return ("C" if result[0] < 0.5 else "D") + "\n"