from flask import Flask, jsonify, request, send_from_directory


app = Flask(__name__)


@app.route('/')
def get_home():
    return send_from_directory('ui', 'Canvas.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
