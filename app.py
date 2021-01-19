import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = tf.keras.models.load_model('models/model.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
    
    path = os.path.join(file_path)
    print(path)
    img = tf.keras.preprocessing.image.load_img(path, target_size=(250, 250))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])
    final=np.argmax(score)
    predictions = 'पीकाचू' if final==0 else 'राइचू'
    if predictions=='पीकाचू':
        color = 'danger'
    else: 
        color = 'warning'
    return render_template('index.html', prediction_text='ये {} है !!' .format(predictions), color = '{}' .format(color))

if __name__ == '__main__':
    app.run(debug=True)