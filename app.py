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

def draw_boxes(image, borders, label, 
               score, h=0.06, fontScale=0.0007):
    '''
    Function to draw the boxes with openCV on a given
    image:

    Input
        image : image where the boxes are plotted
        borders : the borders of the box
        label : the name of the object identified
        score : the confidence of the classification
        h : the height of the boxes
        fontScale : the fontscale for the text

    Output:
        image : the image with the box
    '''
    #Determine the optimal parameters for the image
    displacement = 3
    if borders[0]-h < 0:
        h *= -1
        displacement = 2

    x_max = round(image.shape[1]*borders[3])
    x_min = round(image.shape[1]*borders[1])
    y_max = round(image.shape[0]*borders[2])
    y_min = round(image.shape[0]*borders[0])
    
    h = round(image.shape[0]*h)
    
    image = np.array(image)
    # Add image boxes
    image = cv2.rectangle(image, (x_min, y_min),
                         (x_max, y_max), (0, 255, 0), 2)

    # Add text and boxes
    text = f"{label} {score:.1f}"
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = max(image.shape)*fontScale
    thickness = 1
    labelSize = cv2.getTextSize(text, fontFace, fontScale, thickness)

    image = cv2.rectangle(image, (x_min, y_min - h),
                         (x_min + labelSize[0][0], y_min), (0,255,0), -1)

    image = cv2.putText(image, text,
                        (x_min, y_min - int(h/displacement)),
                        fontFace, fontScale, (0,0,0), thickness)

    return image
    
def predict_objects(image_file : Image.Image, threshold=0.6):
    '''
    Function to classify the object in a given image.
    This function uses a neural network trained on the
    COCO dataset (https://cocodataset.org/#home) 

    Input:
        image : the image to be classified
        threshold : the confidence threshold to make a
                    classification

    Output:
        image : the image with the identified objects

    '''
    basewidth = 1080
    wpercent = (basewidth/float(image_file.size[0]))
    hsize = int((float(image_file.size[1])*float(wpercent)))
    image_file = image_file.resize((basewidth,hsize), Image.ANTIALIAS)

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
    '''
    Homepage of the application. This function
    will render the HTML template in the templates
    folder and will allow the user to upload and generate
    predictions on a given image
    '''
    if request.method == "POST":
        image_file = request.files["image"]
        #If an image is uploaded:
        if image_file:
            data = io.BytesIO() #the image is never stored in a physical disk
            image = predict_objects(Image.open(image_file))            
            Image.fromarray(image).save(data, "JPEG")
            encoded_data = base64.b64encode(data.getvalue())
            return render_template("index.html", 
                                   img_data=encoded_data.decode('utf-8'))

    return render_template("index.html", img_data=None)

if __name__ == "__main__":

    app.run(port=12000, debug=True)

