from flask import Flask
from flask import request

app = Flask(__name__)

@app.post("/sendimages")
def get_data():
    return request.json