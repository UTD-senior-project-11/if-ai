from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)  # 3 channels for RGB

    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    return img


# load an image and predict the class
def evaluate():
    img = load_image('images/google/cat4.jpg')
    # img = load_image('images/google/dog4.jpg')
    model = load_model('final_model.h5')
    result = model.predict(img)
    print("Classification: " + str(result[0]))
    print("Allowed (cat)" if result[0] < 0.5 else "Not allowed (dog)")


evaluate()
