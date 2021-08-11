from utils import draw_objects

from flask import Flask
from flask import render_template, redirect, url_for
from flask import request

from PIL import Image
import base64
import io
import os

app = Flask(__name__)
data = io.BytesIO()

@app.route("/", methods=["GET", "POST"])
def index():

    '''
    Homepage of the app, offer the
    possibility to upload and predict an
    image
    '''
    if request.method == "POST":
        image_file = request.files["fileUpload"]
        #If an image is uploaded:
        if image_file:
            #We convert in RGB (for images with a colormap)
            image = Image.open(image_file).convert('RGB')
            image = draw_objects(image)
            #Reset the stream object
            data.truncate(0) 
            data.seek(0)
            #Add the image to the stream
            Image.fromarray(image).save(data, "JPEG") #Save image in memory

            return redirect(url_for('predict'))   
        
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    '''
    Prediction section of the app, offers a
    button to go to the homepage and the outputs
    from the neural network
    '''
    #Return to the homepage
    if request.method == "POST":
        return redirect(url_for("index"))

    #Render the image
    encoded_data = base64.b64encode(data.getvalue())
    return render_template("predictions.html",
            img_data=encoded_data.decode('utf-8'))

if __name__ == "__main__":

    app.run(debug=False, host="0.0.0.0", 
            port=int(os.environ.get("PORT", 8080)))

