from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model('my_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('L')
    image = image.resize((28, 28))
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    prediction = model.predict(image).argmax()
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
