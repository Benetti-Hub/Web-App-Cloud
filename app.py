from werkzeug.utils import send_file
from utils import draw_objects

from flask import Flask
from flask import render_template, redirect, url_for
from flask import request

from PIL import Image
import base64
import io
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    '''
    Homepage of the app, offer the
    possibility to upload and predict an
    image
    '''
    if request.method == "POST":
        if "fileUpload" in request.files.keys():
            image_file = request.files["fileUpload"]
            #If an image is uploaded:
            #We convert in RGB (for images with a colormap)
            image = Image.open(image_file).convert('RGB')
            image = draw_objects(image)
            #Add the image to the stream
            data = io.BytesIO()
            Image.fromarray(image).save(data, "JPEG") #Save image in memory
            encoded_data = base64.b64encode(data.getvalue())

            return render_template("predictions.html",
                                    img_data=encoded_data.decode('utf-8')) 
        
    return render_template("index.html")

if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", 
            port=int(os.environ.get("PORT", 8080)))

