from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def define_model():
    # load pre-trained model that's built into keras; VGG16 is a very deep CNN with 16 layers
    model = VGG16(include_top=False, input_shape=(224, 224, 3))

    # disable training for all layers
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    layer1 = Flatten()(model.layers[-1].output)
    layer2 = Dense(128, activation='relu', kernel_initializer='he_uniform')(layer1)
    output = Dense(1, activation='sigmoid')(layer2)  # sigmoid for binary classification

    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    # compile model with stochastic gradient descent
    opt = SGD(learning_rate=0.001, momentum=0.9)  # low learning rate for better fine-tuning and stability
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train():
    model = define_model()

    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]  # mean values taken from ImageNet

    train_it = datagen.flow_from_directory('images/combined/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))

    # fit
    model.fit(train_it, steps_per_epoch=len(train_it), epochs=10)
    # save
    model.save('final_model.h5')


train()
