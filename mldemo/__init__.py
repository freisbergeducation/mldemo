# 003_Freisberg_Education\mldemo\mldemo>set FLASK_APP=mldemo
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request, url_for, session
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__, instance_relative_config=True)
app.secret_key = '3ks93k6n4kdilm4jnrkf'

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    #textinput = request.form['textinput']
    #imageinput = request.files['imageinput']
    #img = Image.open(imageinput)
    #new_size = 50
    #factor = new_size/min(img.size)
    
    #img = img.resize((round(factor*img.size[0]), round(factor*img.size[1])), Image.ANTIALIAS)
    
    #if img.size[0]>img.size[1]:
    #  left = round((img.size[0]-new_size)/2)
    #  img = img.crop((left, 0, left+new_size, new_size))
      
    #if img.size[0]<img.size[1]:
    #  upper = round((img.size[1]-new_size)/2)
    #  img = img.crop((0, upper, new_size, upper+new_size))
    
    #img = np.array(img) / 255.0
    #img = np.array([img])
    #prediction = img.shape
    selected_model = request.form.get('selected_model')
    session['selected_model'] = selected_model
    #model = load_model(selected_model + ".h5")
    #class_names = ['buch', 'tasse']
    #prediction = model.predict(img).round(3)
    #prediction = class_names[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
    #img.save("image.jpeg")

    #return render_template("predict.html", prediction=prediction)
    if selected_model == 'tasse_baum_buch':
      return render_template("upload_image.html")

@app.route('/result', methods=['POST', 'GET'])
def result():
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
    #selected_model = request.form.get('selected_model')
    selected_model = session.get('selected_model', None)
    model = load_model(selected_model + ".h5")
    class_names = ['buch', 'tasse']
    prediction = model.predict(img).round(3)
    prediction = class_names[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
    #img.save("image.jpeg")

    return render_template("result.html", prediction=prediction)

# split this up in multiple python files