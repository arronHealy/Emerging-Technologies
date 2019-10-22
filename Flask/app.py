from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import tensorflow as tf
from keras.models import load_model

import keras as kr
import numpy as np
import io
import base64

app = Flask(__name__)

global model, graph


def get_model():
    m = load_model('../jupyter/mnist_num_reader.model')
    print('Model loaded')
    return m


print('Loading model & Graph...')
model = get_model()
graph = tf.get_default_graph()


"""
def preprocess_image(imageData):
    canvasImage = Image.open(io.BytesIO(base64.b64decode(imageData)))
    canvasImage = canvasImage.resize((28, 28), Image.LANCZOS)

    image = Image.new("L", canvasImage.size, (255))
    image.paste(canvasImage, canvasImage)
    image = ImageOps.invert(image)

    return image
"""


@app.route('/')
def get_home():
    return send_from_directory('ui', 'Canvas.html')


@app.route('/classify-image', methods=['POST'])
def classify():
    img_size = (28, 28)

    data = request.get_json()

    encoded = data["imageData"]

    img = Image.open(io.BytesIO(base64.b64decode(encoded)))

    img = img.resize(img_size, Image.ANTIALIAS)

    img = img.convert('1')

    img_array = np.asarray(img)
    img_array = img_array.flatten()

    with graph.as_default():
        out = model.predict(img_array)
        print(out)
        print(np.argmax(out, axis=1))
        response = {
            'prediction': np.array_str(np.argmax(out, axis=1))
        }
        return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
