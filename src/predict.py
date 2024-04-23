from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import img_to_array

import numpy as np
import io
import base64
from PIL import Image

def load_my_image(imgString):
    img = imgString.resize((224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)  # 3 channels for RGB

    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    return img

# load an image and predict the class
def evaluate(imageString):
    img = load_my_image(imageString)
    model = load_model('./src/final_model.h5')
    result = model.predict(testimg)
    return result[0]