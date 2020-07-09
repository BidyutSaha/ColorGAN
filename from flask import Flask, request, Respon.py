from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/colorify', methods=['POST'])
def colorify():
    print("hello world")
    print(request.form, "hello")
    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# start flask app
print("helo world!")
app.run(host="0.0.0.0", port=5000)
