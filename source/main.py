from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import base64
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = tf.keras.models.load_model('mnist_model_optimized.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận ảnh từ canvas
    data_url = request.json['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    img = Image.open(img_bytes).convert('L')

    # Đảo màu (MNIST: nền đen, chữ trắng)
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Tiền xử lý
    arr = np.array(img).astype('float32') / 255.0
    arr = gaussian_filter(arr, sigma=0.5)  # Giảm sigma để giữ chi tiết
    arr = arr.reshape(1, 28, 28, 1)

    # Lưu ảnh đầu vào để kiểm tra
    os.makedirs('input_images', exist_ok=True)
    plt.imsave(f'input_images/input_{len(os.listdir("input_images"))}.png', arr.reshape(28, 28), cmap='gray')

    # Dự đoán
    pred = model.predict(arr, verbose=0)
    digit = int(np.argmax(pred, axis=1)[0])
    probabilities = [float(p) for p in pred[0]]  # Chuyển xác suất thành danh sách

    return jsonify({'digit': digit, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True) 