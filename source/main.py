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

Directory = os.path.dirname(os.path.abspath(__file__))

# app = Flask(__name__
# , templates = 'D:/MNIST - Handwriting Recognition/templates'
# , statics = 'D:/MNIST - Handwriting Recognition/statics')

# có cách nào lấy thư mục hiện tại và sử dụng nó không?

# model = tf.keras.models.load_model('D:/MNIST - Handwriting Recognition/source/model.h5')
    # Nhận ảnh từ canvas

app = Flask(__name__,
            template_folder=os.path.join(Directory, 'templates'),
            static_folder=os.path.join(Directory, 'statics'))
            # assets_folder=os.path.join(Directory, 'assets'))
            # histories_folder=os.path.join(assets_folder, 'histories'))

model = tf.keras.models.load_model(os.path.join(Directory, 'model.h5'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json['image']
    header, encoded = data_url.split(',', 1)
    img_bytes = io.BytesIO(base64.b64decode(encoded))
    image = Image.open(img_bytes).convert('L')

    # Đảo màu (MNIST: nền đen, chữ trắng)
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Tiền xử lý
    arr = np.array(image).astype('float32') / 255.0
    arr = gaussian_filter(arr, sigma=0.5)  # Giảm sigma để giữ chi tiết
    arr = arr.reshape(1, 28, 28, 1)

    # Lưu ảnh đầu vào để kiểm tra
    # os.makedirs('input_images', exist_ok=True)
    # plt.imsave(f'input_images/input_{len(os.listdir("input_images"))}.png', arr.reshape(28, 28), cmap='gray')

    # tôi muốn lưu vào thư mục assets
    os.makedirs(os.path.join(Directory, 'assets', 'histories'), exist_ok=True)
    plt.imsave(os.path.join(Directory, 'assets', 'histories', f'{len(os.listdir(os.path.join(Directory, "assets", "image")))}.png'), arr.reshape(28, 28), cmap='gray')
    
    # Dự đoán
    prediction = model.predict(arr, verbose=0)
    digit = int(np.argmax(prediction, axis=1)[0])
    probabilities = [float(p) for p in prediction[0]]  # Chuyển xác suất thành danh sách

    return jsonify({'digit': digit, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True) 