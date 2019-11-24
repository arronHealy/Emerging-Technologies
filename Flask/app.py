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


# Send Web page when base route is reached

@app.route('/')
def get_home():
    return send_from_directory('ui', 'Canvas.html')


"""
    Post route takes image data as base 64 string and converts back to image to predict what user drew.

    Some code adapted from youtube series on Deep lizard channel:
    https://www.youtube.com/watch?v=SI1hVGvbbZ4&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=22

    and from
    https://heartbeat.fritz.ai/image-classification-on-android-using-a-keras-model-deployed-in-flask-118adffc5045
"""


@app.route('/classify-image', methods=['POST'])
def classify():
    # default mnist image size ready
    img_size = (28, 28)

    # read request json containing image data
    data = request.get_json()

    # assign base 64 string to variable
    encoded = data["dataUrl"]

    # convert image and save as png
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(encoded))

    # read saved image in as grayscale
    x = imread('output.png', mode='L')

    # resize image to mnist size
    x = imresize(x, img_size)

    # convert and scale image data as in model preperation
    x = np.array(x, dtype=np.float32).reshape(1, 784)

    x /= 255

    print('x reshape', x.shape)

    # predict using temsorflow default graph
    # https://stackoverflow.com/questions/56376943/running-keras-predictions-with-flask-gives-error

    with graph.as_default():
        predictions = list(model.predict(x))

    # print('\nresults: ', str(predictions))

    # print('prediction 0 ', predictions[0])

    print('prediction is ', np.argmax(predictions))

    prediction = {
        'prediction': 'Neural Network predicts: {}'.format(np.argmax(predictions))
    }

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
