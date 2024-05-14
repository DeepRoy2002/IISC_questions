from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('C:/Users/anirb/IISC questions/cifar10/model_1.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((32, 32))  # Resize the image to match the input size of your model
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            image = Image.open(file)
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_class = class_labels[np.argmax(prediction)]
            return render_template('result.html', predicted_class=predicted_class)
    
    return render_template('index.html')

@app.route('/url_predict', methods=['POST'])
def url_predict():
    image_url = request.form['image_url']
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        return render_template('result.html', predicted_class=predicted_class)
    except Exception as e:
        return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)
