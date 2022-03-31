# 003_Freisberg_Education\mldemo\mldemo>set FLASK_APP=mldemo
# set FLASK_ENV=development
# flask run

from flask import Flask, render_template, request, url_for, session
from PIL import Image
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
import gc
from tensorflow import keras
import tensorflow as tf
import pickle
from pydub import AudioSegment
from random import randint
import os
import time

app = Flask(__name__, instance_relative_config=True)
app.secret_key = '3ks93k6n4kdilm4jnrkf'

@app.route('/')
def index():
  return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
  image_models = [
    'Gesten_Gruppe_1',
    'Emotionen_Gruppe_2',
    'Emotionen2_Gruppe_2',
    'Knoten_Gruppe_5',
    "Catdog_Gruppe_3"
  ]
  audio_models = [
    'Sprachen_Gruppe_4'
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
    'Gesten_Gruppe_1': ['open', 'closed'],
    'Emotionen_Gruppe_2': ['happy', 'sad', 'angry', 'poker_face', 'bored'],
    'Emotionen2_Gruppe_2': ['gluecklich', 'gelangweilt', 'verärgert'],
    'Knoten_Gruppe_5': ['Doppelter Achter', 'Prusikknoten', 'Ueberhandknoten'],
    "Catdog_Gruppe_3": ['dogs', 'Realcats', 'maxibanane', 'snoopdogg']
  }
  image_input = request.files['image_input']
  img = Image.open(image_input)

  selected_model = session.get('selected_model', None)

  if selected_model == "Gesten_Gruppe_1":
    new_size = 70
  elif selected_model == "Knoten_Gruppe_5":
    new_size = 100
  else:
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
  model = keras.models.load_model("models/" + selected_model + ".h5")
  labels = labels_dict[selected_model]
  prediction = model.predict(img).round(3)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
  del model
  gc.collect()

  return render_template("result.html", prediction=prediction)

@app.route('/result_text', methods=['POST'])
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
  vectorize_layer_config = pickle.load(open("models/" + selected_model + "_vectorize_layer.pkl", "rb"))
  vectorize_layer = keras.layers.TextVectorization.from_config(vectorize_layer_config['config'])
  vectorize_layer.set_weights(vectorize_layer_config['weights'])
  text = vectorize_layer(text)
  model = keras.models.load_model("models/" + selected_model + ".h5")
  labels = labels_dict[selected_model]
  prediction = model.predict(text).round(3)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
  del model
  gc.collect()

  return render_template("result.html", prediction=prediction)

@app.route('/result_audio', methods=['POST'])
def result_audio():
  labels_dict = {
    'Sprachen_Gruppe_4': ["noise", "de", "en"]
  }

  input_len = 22050
  audio_input = request.files['audio_input']
  random_nr = randint(1000, 9999)
  audio_file_path = os.path.join('audio/', 'audio_input_' + str(random_nr) + '.m4a')
  audio_file_path_wav = os.path.join('audio/', 'audio_input_' + str(random_nr) + '.wav')
  app.logger.info("######################## 1 ############################")
  audio_input.save(audio_file_path)

  app.logger.info(os.path.exists(audio_file_path))

  while not os.path.exists(audio_file_path):
    time.sleep(1)

  app.logger.info(os.path.exists(audio_file_path))
  audio_input = AudioSegment.from_file(audio_file_path, format= 'm4a')
  audio_input.export(audio_file_path_wav, format="wav")

  audio_input = tf.io.read_file("./audio/audio_input_" + str(random_nr) + ".wav")
  audio_input, _ = tf.audio.decode_wav(contents=audio_input)
  audio_input = tf.squeeze(audio_input, axis=-1)
  audio_input = audio_input[:input_len]
  zero_padding = tf.zeros(
      [input_len] - tf.shape(audio_input),
      dtype=tf.float32)
  audio_input = tf.cast(audio_input, dtype=tf.float32)
  equal_length = tf.concat([audio_input, zero_padding], 0)
  audio_input = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  audio_input = tf.abs(audio_input)
  audio_input = audio_input[..., tf.newaxis]
  audio_input = np.asarray([audio_input])
  
  selected_model = session.get('selected_model', None)
  model = keras.models.load_model("models/" + selected_model + ".h5")
  labels = labels_dict[selected_model]
  prediction = model.predict(audio_input).round(3)
  app.logger.info(prediction)
  prediction = labels[np.argmax(prediction[0])] + " (" + str(int(round(100*max(prediction[0])))) +"%)"
  del model
  del audio_input
  gc.collect()
  os.remove(audio_file_path)
  os.remove(audio_file_path_wav)

  return render_template("result.html", prediction=prediction)

# split this up in multiple python files