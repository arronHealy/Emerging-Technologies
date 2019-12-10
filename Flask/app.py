from flask import Flask, jsonify, request, send_from_directory

# from imageio import imread, imresize
from PIL import Image

import tensorflow as tf
from keras.models import load_model
from keras import backend as K

import keras as kr
import numpy as np
import io
import base64

app = Flask(__name__)

global model, graph


def get_model(model):
    if model == 'sequential':
        m = load_model('../jupyter/mnist_num_reader.h5')
    else:
        m = load_model('../jupyter/mnist_conv2d.h5')

    print('Model loaded')
    return m


# print('Loading model & Graph...')
# model = get_model()
# graph = tf.get_default_graph()


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
    # x = imread('output.png', mode='L')
    x = Image.open('output.png').convert('L')

    # resize image to mnist size
    x = x.resize(img_size, Image.ANTIALIAS)

    # load model based upon model specified
    # model being loaded in same thread as prediction fixs issue described below
    model = get_model(data["model"])

    # reshape depending on model specified
    # convert and scale image data as in model preperation
    if data["model"] == 'sequential':
        x = np.array(x, dtype=np.float32).reshape(1, 784) / 255
    else:
        x = np.array(x, dtype=np.float32).reshape(1, 28, 28, 1) / 255

    print('x shape', x.shape)

    # model.predict() wont't work on it's own
    # had to use the following
    # predict using tensorflow default graph
    # https://stackoverflow.com/questions/53874115/keras-model-working-fine-locally-but-wont-work-on-flask-api

    # with graph.as_default():

    # figured out the issue since model was being loaded on separate thread it wouldn't work
    # but now model being loaded in same thread so predict works fine
    predictions = list(model.predict(x))

    # print('\nresults: ', str(predictions))

    # print('prediction 0 ', predictions[0])

    print('prediction is ', np.argmax(predictions))

    prediction = {
        'prediction': 'Neural Network predicts: {}'.format(np.argmax(predictions))
    }

    # clear backend session so not to cause crash if different model specified
    # https://stackoverflow.com/questions/51588186/keras-tensorflow-typeerror-cannot-interpret-feed-dict-key-as-tensor
    K.clear_session()

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
