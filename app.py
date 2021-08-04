from utils import di

from flask import Flask
from flask import render_template
from flask import request

import base64
import io

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import cv2
from PIL import Image

app = Flask(__name__)
model = hub.load("model/")

UPLOAD_FOLDER = "./static"

def draw_boxes(image, borders, label :str, 
               score :float, h=0.02, fontScale=0.0004):
    
    if borders[0]-h < 0:
        h *= -1

    x_max = round(image.shape[1]*borders[3])
    x_min = round(image.shape[1]*borders[1])
    y_max = round(image.shape[0]*borders[2])
    y_min = round(image.shape[0]*borders[0])
    
    h = round(image.shape[0]*h)
    fontScale = image.shape[1]*fontScale
    
    image = cv2.rectangle(image, (x_min, y_min),
                         (x_max, y_max), (0, 255, 0), 2)

    image = cv2.rectangle(image, (x_min, y_min-h),
                          (x_max, y_min), (0,255,0), -1)

    # Add text
    image = cv2.putText(image, (f"{label} {score:.1f}"),
                        (x_min, y_min - int(h/3)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (0,0,0), 1)

    return image
    
def predict_objects(image_file, threshold=0.6):

    image = np.array(image_file)[..., :3]
    image_nc = image[np.newaxis, ...]
    img_tensor = tf.convert_to_tensor(
                    image_nc, dtype=tf.uint8
                 )

    detector_output = model(img_tensor)
    
    for i in range(int(detector_output["num_detections"])):

        boxes = detector_output['detection_boxes'][0][i].numpy()
        score = detector_output['detection_scores'][0][i].numpy()
        label = di[detector_output['detection_classes'][0][i].numpy()]['name']
        
        if score > threshold:
            image = draw_boxes(image, boxes, label, 100*score)

    return image


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            data = io.BytesIO()
            image = predict_objects(Image.open(image_file))            
            Image.fromarray(image).save(data, "JPEG")
            encoded_data = base64.b64encode(data.getvalue())
            return render_template("index.html", 
                                   img_data=encoded_data.decode('utf-8'))

    return render_template("index.html", img_data=0)

if __name__ == "__main__":

    app.run(port=12000, debug=True)

