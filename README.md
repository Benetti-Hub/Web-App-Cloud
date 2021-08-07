# Web-App-Cloud (IN PROGRESS)
This project showcases the code for the deployement of a simple web-app using containers with Google App Run, allowing for Continuos Delivery (CD) and automated scalability of the application. 

The application take as an imput an image, and uses a neural network trained on the COCO 2017 dataset to perform inference (taken from TensorflowHub \href{https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2}). 

## No image is stored anywhere at anytime
In fact, if you look at the source code, to print the image, it is converted in binary and never stored in a google cloud storage bucket.

