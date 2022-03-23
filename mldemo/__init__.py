# 003_Freisberg_Education\mldemo\mldemo>set FLASK_APP=mldemo
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request, url_for, session
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
from tensorflow import keras

app = Flask(__name__, instance_relative_config=True)
app.secret_key = '3ks93k6n4kdilm4jnrkf'

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/upload', methods=['POST', 'GET'])
def upload():
  image_models = [
    'buch_oder_tasse'
  ]
  audio_models = [
    'zero_or_one'
  ]
  text_models = [
    'hate_speech'
  ]
  selected_model = request.form.get('selected_model')
  session['selected_model'] = selected_model

  if selected_model in image_models:
    return render_template("upload_image.html")
  elif selected_model in audio_models:
    return render_template("upload_audio.html")
  elif selected_model in text_models:
    return render_template("upload_text.html")

@app.route('/result_image', methods=['POST', 'GET'])
def result_image():
  labels_dict = {
    'buch_oder_tasse': ['buch', 'tasse']
  }
  image_input = request.files['image_input']
  img = Image.open(image_input)
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
  selected_model = session.get('selected_model', None)
  model = keras.models.load_model("models/" + selected_model)
  labels = labels_dict[selected_model]
  prediction = model.predict(img).round(3)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"

  return render_template("result.html", prediction=prediction)

""" @app.route('/result_text', methods=['POST', 'GET'])
def result_text():
  labels_dict = {
    'hate_speech': ['hate', 'no hate']
  }
  #nltk.download('stopwords')
  text_input = request.form['text_input']

  stemmer = PorterStemmer()
  stopwords_german = stopwords.words('german')

  # tokenize text
  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                              reduce_len=True)
  text_tokens = tokenizer.tokenize(text_input)
  text_clean = []
  for word in text_tokens:
      if (word not in stopwords_german and
              word not in string.punctuation):
          stem_word = stemmer.stem(word)
          text_clean.append(stem_word)

  text = np.asarray([[' '.join(text_clean)]])

  selected_model = session.get('selected_model', None)
  model = keras.models.load_model("models/" + selected_model)
  labels = labels_dict[selected_model]
  prediction = model.predict(text).round(3)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"

  return render_template("result.html", prediction=prediction)

@app.route('/result_audio', methods=['POST', 'GET'])
def result_audio():
  labels_dict = {
    'hate_speech': ['zero', 'one']
  }
  
  selected_model = session.get('selected_model', None)
  model = keras.models.load_model("models/" + selected_model)
  labels = labels_dict[selected_model]
  audio = ""
  prediction = model.predict(audio).round(3)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"

  return render_template("result.html", prediction=prediction) """

# split this up in multiple python files