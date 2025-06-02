from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

#* Flask: Python web framework
#   - Input: resquest 
#   - Output: response (jsonify)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

source_folder = os.path.dirname(os.path.abspath(__file__))
Directory = os.path.dirname(source_folder)


app = Flask(__name__,
            template_folder=os.path.join(Directory, 'template'),
            static_folder=os.path.join(Directory, 'static'))

print(os.path.join(source_folder, 'model.h5'))
model = tf.keras.models.load_model(os.path.join(source_folder, 'model.h5'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    image = Image.open(img_bytes).convert('L')

    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(image).astype('float32') / 255.0
    arr = gaussian_filter(arr, sigma=0.5)  # Giảm sigma để giữ chi tiết
    arr = arr.reshape(1, 28, 28, 1)



    os.makedirs(os.path.join(Directory, 'assets', 'histories'), exist_ok=True)
    plt.imsave(os.path.join(Directory, 'assets', 'histories', f'{len(os.listdir(os.path.join(Directory, "assets", "histories")))}.png'), arr.reshape(28, 28), cmap='gray')
    
    prediction = model.predict(arr, verbose=0)
    digit = int(np.argmax(prediction, axis=1)[0])
    probabilities = [float(p) for p in prediction[0]]  # Chuyển xác suất thành danh sách

    return jsonify({'digit': digit, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True) 