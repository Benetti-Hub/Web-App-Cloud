# Web-App-Cloud (IN PROGRESS)
This project showcases the code for the deployement of a simple web-app using containers with Google App Run, allowing for Continuos Delivery (CD) and automated scalability of the application. 

The application take as an imput an image, and uses a neural network trained on the COCO dataset (https://cocodataset.org/#home) to perform inference (taken from TensorflowHub (https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2). 

## No image is stored anywhere at anytime
You can check this since the code is open source. A connection to a google cloud storage bucket would be trivial, but given than anyone can post an image, I did not want to fill my google cloud account with random files.
