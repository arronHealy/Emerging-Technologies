from flask import Flask, jsonify, request, send_from_directory

from scipy.misc import imread, imresize

import tensorflow as tf
from keras.models import load_model

import keras as kr
import numpy as np
import io
import base64

app = Flask(__name__)

global model, graph


def get_model():
    m = load_model('../jupyter/mnist_num_reader.h5')
    print('Model loaded')
    return m


print('Loading model & Graph...')
model = get_model()
graph = tf.get_default_graph()


@app.route('/')
def get_home():
    return send_from_directory('ui', 'Canvas.html')


@app.route('/classify-image', methods=['POST'])
def classify():
    img_size = (28, 28)

    data = request.get_json()

    encoded = data["dataUrl"]

    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(encoded))

    x = imread('output.png', mode='L')

    x = imresize(x, img_size)

    # print(x)

    x = np.array(x, dtype=np.float32).reshape(1, 784)

    x /= 255

    print('x reshape', x.shape)

    with graph.as_default():
        predictions = list(model.predict(x))

    print('\nresults: ', str(predictions))

    print('prediction 0 ', predictions[0])

    print('\nprediction is ', np.argmax(predictions))

    return jsonify({'prediction': 'Neural Network predicts: {}'.format(np.argmax(predictions))})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
