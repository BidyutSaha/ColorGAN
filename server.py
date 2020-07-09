import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
import cv2
from flask_cors import CORS
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import re
from skimage import color
import matplotlib.pyplot as plt


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


def l2(y_true, y_pred):

    x = K.mean(K.square(y_true-y_pred), axis=-1)
    return x


def l1(y_true, y_pred):

    x = K.mean(K.abs(y_true-y_pred), axis=-1)
    return x


def l3(y_true, y_pred):

    x = K.mean(K.sqrt(K.sum(K.square(y_true-y_pred), axis=-1)))
    return x


custom_losses = {
    "lossl1": l1,
    "lossl2": l2,
    "euclidean_distance_loss": l3,
    "l3": l3

}


aiColorGenerator = load_model(
    "./lib/aicolorgen.h5", custom_objects=custom_losses)


# Initialize the Flask application
app = Flask(__name__)
CORS(app)


def numpyToPNGB64(arr):

    im = Image.fromarray(arr.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    img_str = "data:image/png;base64," + \
        base64.b64encode(rawBytes.read()).decode("utf-8")

    return img_str

# route http posts to this method
@app.route('/api/colorify', methods=['POST'])
def colorify():
    # filtering only the content part from base64 image string
    img = re.search(r'base64,(.*)',
                    request.get_json(force=True)["gray_img"]).group(1)

    base64_decoded = base64.b64decode(img)

    # image in rgb
    image = Image.open(io.BytesIO(base64_decoded)).convert(
        "RGB").resize((256, 256), Image.ANTIALIAS)
    image_np = np.array(image)

    # rgb to  lab conversion
    lab = color.rgb2lab(image_np)

    scalled_lab = lab/100
    l = scalled_lab[..., 0]

    ref_img = np.dstack((l, l, l))*100*2.5

    p = aiColorGenerator.predict(
        scalled_lab[..., 0].reshape((1, 256, 256, 1)))[0]

    predicted_img = np.dstack(
        (l, p[..., 0], p[..., 1]))*100

    processed_img = color.lab2rgb(predicted_img)*255
    print(processed_img.min(), processed_img.max())

    ref_img_str = numpyToPNGB64(ref_img)
    img_str = numpyToPNGB64(processed_img)

    response = jsonify({'img1': img_str, 'img2': ref_img_str})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# start flask app
app.run(host="0.0.0.0", port=5000)


# https://stackoverflow.com/questions/46598607/how-to-convert-a-numpy-array-which-is-actually-a-bgr-image-to-base64-string/46599592
