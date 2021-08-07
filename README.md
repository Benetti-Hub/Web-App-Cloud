# Web-App-Cloud (IN PROGRESS)
This project showcases the code for the deployement of a simple web-app using containers with Google App Run, allowing for Continuos Delivery (CD) and automated scalability of the application. 

The application take as an imput an image, and uses a neural network trained on the COCO dataset (https://cocodataset.org/#home) to perform inference (taken from TensorflowHub: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2). 

## No image is stored anywhere at anytime
You can easily check this claim since the code is open source. A connection to a google cloud storage bucket would be trivial, but given than anyone can post an image, I did not want to fill my google cloud account with random files.

## This app is only for my GitHub portfolio  
Since most of my projects are currently private (I am waiting to be published on Elsevier), I decided to create this app to show that I can actually work with Python. Given that no restriction is imposed on the type of files one can upload, it is evident that bugs might be present. In particualr, this application was tested mainly against JPEG and PNG files, other extensions might cause an error. Again, the scope of this project is to deploy the application, not to create a very complex and performing machine learning model (I can do the latter, I swear, but it would require much more time).
