from flask import Flask, jsonify, request, send_from_directory
from scipy.misc import imread, imresize, imsave
import tensorflow as tf
from keras.models import load_model
import numpy as np
import io
import base64

app = Flask(__name__)

model = load_model('../jupyter/mnist_num_reader.model')

graph = tf.get_default_graph()


@app.route('/')
def get_home():
    return send_from_directory('ui', 'Canvas.html')


@app.route('/classify-image', methods=['POST'])
def classify():
    img_size = (28, 28)

    data = request.get_json()

    img = base64.b64decode(data['imageData'])

    # x = imresize(img, (28, 28)) / 255

    # x = x.reshape(1, 28, 28)

    # image_file = 'test.png'

    # with open(image_file, 'wb') as f:
    # f.write(img)


"""
    with graph.as_default():
        result = model.predict(x)
        data = {
            'results': result
        }
        return jsonify(response)
        """


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
