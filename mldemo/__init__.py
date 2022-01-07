from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__, instance_relative_config=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    #textinput = request.form['textinput']
    imageinput = request.files['imageinput']
    img = Image.open(imageinput)
    new_size = 50
    factor = new_size/min(img.size)
    
    img = img.resize((round(factor*img.size[0]), round(factor*img.size[1])), Image.ANTIALIAS)
    
    if img.size[0]>img.size[1]:
      left = round((img.size[0]-new_size)/2)
      img = img.crop((left, 0, left+new_size, new_size))
      
    if img.size[0]<img.size[1]:
      upper = round((img.size[1]-new_size)/2)
      img = img.crop((0, upper, new_size, upper+new_size))
    
    img = np.array(img) / 255.0
    img = np.array([img])
    #prediction = img.shape
    model = load_model("tasse_baum_buch.h5")
    class_names = ['buch', 'tasse', 'baum']
    prediction = model.predict(img).round(3)
    prediction = class_names[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
    #img.save("image.jpeg")

    return render_template("index.html", prediction=prediction)

# split this up in multiple python files