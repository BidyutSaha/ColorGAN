
# importing diffrent utility modules and frameworks
import tensorflow as tf
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
from flask import render_template
from flask_cachebuster import CacheBuster


# Handeling the Shaered GPU uses by diffrent process in the Same host
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# defining the user defined loss function for AI agent
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


# defining the AI agent to colorize the given grayscale image
aiColorGenerator = load_model(
    "./lib/aicolorgen.h5", custom_objects=custom_losses)


# Initialize the Flask application
app = Flask(__name__, static_url_path="",
            template_folder='static', static_folder="static")


# handeling the CROS Browser Error in REST API invocation
CORS(app)

# image processing pipeline for generating the colored version of given gray scale image
# [grayscale--->lab---->L---->scalled L-----> predict--------->reverse scaleof AB---->LAB--->multiply 100--->lab2rgb---->multiply 255]


def numpyToPNGB64(arr):
    """[Converts a numpy image to png formated image in base 64 reperesentation]

    Args:
        arr ([numpy]): [image in numpy format]

    Returns:
        [string]: [base64 version of the given numpy image]
    """

    im = Image.fromarray(arr.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    img_str = "data:image/png;base64," + \
        base64.b64encode(rawBytes.read()).decode("utf-8")

    return img_str


def stored_samleImgColorify(imgId):
    """[retrive the sample image from sample database and returns the image in numpy array]

    Args:
        imgId ([number]): [id of the image]

    Returns:
        [numpy]: [numpy image]
    """
    path = "./static/sampleImg/sample{}.png".format(imgId)
    image_np = np.asarray(Image.open(path))
    return image_np


def numpyGrayToLRefBase64(grayImg_np):
    """[convert a single l channel (from LAB color space) image into RGB formated grayscale image]

    Args:
        grayImg_np ([numpy]): [grayscale image]

    Returns:
        [string]: [base64 version of RGB formated grayscale image]
    """

    lab = color.rgb2lab(grayImg_np)
    scalled_lab = lab/100
    l = scalled_lab[..., 0]
    ref_img = np.dstack((l, l, l))*100*2.5
    img_str = numpyToPNGB64(ref_img)
    return img_str


def numpyGrayToColorBase64(grayImg_np):
    """[convert a numpy grayscasle image to a color version using A.I. Agent]

    Args:
        grayImg_np ([numpy]): [grayscale image in RGB format]

    Returns:
        [string]: [ai colored image in RGB formatted in base64 represented]
    """

    lab = color.rgb2lab(grayImg_np)
    scalled_lab = lab/100
    l = scalled_lab[..., 0]

    p = aiColorGenerator.predict(
        scalled_lab[..., 0].reshape((1, 256, 256, 1)))[0]

    predicted_img = np.dstack(
        (l, p[..., 0], p[..., 1]))*100

    processed_img = color.lab2rgb(predicted_img)*255
    img_str = numpyToPNGB64(processed_img)
    return img_str


@app.route("/", methods=["GET"])
def landing():
    return render_template("index.html")


# route http posts to this method
@app.route('/api/sampleColorify', methods=['POST'])
def sampleColorify():
    imgId = request.get_json(force=True)["gray_img_id"]
    image_np = stored_samleImgColorify(imgId)
    img_str = numpyGrayToColorBase64(image_np)
    ref_img_str = numpyGrayToLRefBase64(image_np)
    response = jsonify({'img1': img_str, 'img2': ref_img_str})
    return response


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

    img_str = numpyGrayToColorBase64(image_np)
    ref_img_str = numpyGrayToLRefBase64(image_np)

    response = jsonify({'img1': img_str, 'img2': ref_img_str})
    return response


# start flask app
app.run(host="0.0.0.0", port=5000, debug=True)
