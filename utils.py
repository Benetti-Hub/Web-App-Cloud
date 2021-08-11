import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import cv2
from PIL import Image, ImageOps, ExifTags

#TensorflowHub Model
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


def draw_objects(image_file : Image.Image, threshold=0.6):
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
    image_file = ImageOps.exif_transpose(image_file)
    basewidth = 512
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

# COCO 2017 dictionary to associate classes to real objects
di = {1: {'id': 1, 'name': 'person'},
 2: {'id': 2, 'name': 'bicycle'},
 3: {'id': 3, 'name': 'car'},
 4: {'id': 4, 'name': 'motorcycle'},
 5: {'id': 5, 'name': 'airplane'},
 6: {'id': 6, 'name': 'bus'},
 7: {'id': 7, 'name': 'train'},
 8: {'id': 8, 'name': 'truck'},
 9: {'id': 9, 'name': 'boat'},
 10: {'id': 10, 'name': 'traffic light'},
 11: {'id': 11, 'name': 'fire hydrant'},
 13: {'id': 13, 'name': 'stop sign'},
 14: {'id': 14, 'name': 'parking meter'},
 15: {'id': 15, 'name': 'bench'},
 16: {'id': 16, 'name': 'bird'},
 17: {'id': 17, 'name': 'cat'},
 18: {'id': 18, 'name': 'dog'},
 19: {'id': 19, 'name': 'horse'},
 20: {'id': 20, 'name': 'sheep'},
 21: {'id': 21, 'name': 'cow'},
 22: {'id': 22, 'name': 'elephant'},
 23: {'id': 23, 'name': 'bear'},
 24: {'id': 24, 'name': 'zebra'},
 25: {'id': 25, 'name': 'giraffe'},
 27: {'id': 27, 'name': 'backpack'},
 28: {'id': 28, 'name': 'umbrella'},
 31: {'id': 31, 'name': 'handbag'},
 32: {'id': 32, 'name': 'tie'},
 33: {'id': 33, 'name': 'suitcase'},
 34: {'id': 34, 'name': 'frisbee'},
 35: {'id': 35, 'name': 'skis'},
 36: {'id': 36, 'name': 'snowboard'},
 37: {'id': 37, 'name': 'sports ball'},
 38: {'id': 38, 'name': 'kite'},
 39: {'id': 39, 'name': 'baseball bat'},
 40: {'id': 40, 'name': 'baseball glove'},
 41: {'id': 41, 'name': 'skateboard'},
 42: {'id': 42, 'name': 'surfboard'},
 43: {'id': 43, 'name': 'tennis racket'},
 44: {'id': 44, 'name': 'bottle'},
 46: {'id': 46, 'name': 'wine glass'},
 47: {'id': 47, 'name': 'cup'},
 48: {'id': 48, 'name': 'fork'},
 49: {'id': 49, 'name': 'knife'},
 50: {'id': 50, 'name': 'spoon'},
 51: {'id': 51, 'name': 'bowl'},
 52: {'id': 52, 'name': 'banana'},
 53: {'id': 53, 'name': 'apple'},
 54: {'id': 54, 'name': 'sandwich'},
 55: {'id': 55, 'name': 'orange'},
 56: {'id': 56, 'name': 'broccoli'},
 57: {'id': 57, 'name': 'carrot'},
 58: {'id': 58, 'name': 'hot dog'},
 59: {'id': 59, 'name': 'pizza'},
 60: {'id': 60, 'name': 'donut'},
 61: {'id': 61, 'name': 'cake'},
 62: {'id': 62, 'name': 'chair'},
 63: {'id': 63, 'name': 'couch'},
 64: {'id': 64, 'name': 'potted plant'},
 65: {'id': 65, 'name': 'bed'},
 67: {'id': 67, 'name': 'dining table'},
 70: {'id': 70, 'name': 'toilet'},
 72: {'id': 72, 'name': 'tv'},
 73: {'id': 73, 'name': 'laptop'},
 74: {'id': 74, 'name': 'mouse'},
 75: {'id': 75, 'name': 'remote'},
 76: {'id': 76, 'name': 'keyboard'},
 77: {'id': 77, 'name': 'cell phone'},
 78: {'id': 78, 'name': 'microwave'},
 79: {'id': 79, 'name': 'oven'},
 80: {'id': 80, 'name': 'toaster'},
 81: {'id': 81, 'name': 'sink'},
 82: {'id': 82, 'name': 'refrigerator'},
 84: {'id': 84, 'name': 'book'},
 85: {'id': 85, 'name': 'clock'},
 86: {'id': 86, 'name': 'vase'},
 87: {'id': 87, 'name': 'scissors'},
 88: {'id': 88, 'name': 'teddy bear'},
 89: {'id': 89, 'name': 'hair drier'},
 90: {'id': 90, 'name': 'toothbrush'}}
