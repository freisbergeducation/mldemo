from flask import Flask, render_template, request, url_for
from PIL import Image

app = Flask(__name__, instance_relative_config=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    textinput = request.form['textinput']
    imageinput = request.files['imageinput']
    img = Image.open(imageinput)
    img = img.save("image.jpeg")
    return render_template("index.html", textinput=textinput, imageinput=imageinput)

# split this up in multiple python files