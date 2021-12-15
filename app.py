from re import template
from flask import Flask, render_template, request
from cv2 import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/after', methods = ['POST'])
def after():
    img = request.files['file1']

    img.save('static/file.jpg')

    image = cv2.imread('static/file.jpg',0)

    image = cv2.resize(image, (48,48))

    image = np.reshape(image, (48,48,3))

    model = load_model('model.h5')

    prediction = model.predict(image)

    #if prediction[0] == 0:
        #product = ""
    #elif

    label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]
    print(final_prediction)



    return render_template('after.html', data= final_prediction )


if __name__ == "__main__":
    app.run(debug = True) 