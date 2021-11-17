import tensorflow as tf
import tensorflow_hub as hub
import json


IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def getClassNames(json_file):
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    return class_names


def loadModel(filepath_saved_model):
    loaded_model = tf.keras.models.load_model(filepath_saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
    loaded_model.summary()
    return loaded_model

def processImage(im):
    # Converting to tensor
    im = tf.convert_to_tensor(im, dtype = tf.float32)
    # Resizing the image
    im = tf.image.resize(im, (IMG_SIZE, IMG_SIZE))
    # Normalizing the output
    im /= 255
    return im.numpy()