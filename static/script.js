const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const deleteBtn = document.getElementById('deleteBtn');
const eraseBtn = document.getElementById('eraseBtn');
const predictBtn = document.getElementById('predictBtn');
let isDrawing = false;
let isErasing = false;
let debounceTimer;

// Cấu hình canvas
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';

// Xử lý sự kiện chuột
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Xử lý sự kiện cảm ứng
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e.touches[0]);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e.touches[0]);
});
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    resetDebounceTimer();
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    resetDebounceTimer();
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    resetDebounceTimer();
}

// Chuyển đổi chế độ vẽ/tẩy
eraseBtn.addEventListener('click', () => {
    isErasing = !isErasing;
    ctx.strokeStyle = isErasing ? 'white' : 'black';
    eraseBtn.textContent = isErasing ? 'Draw' : 'Erase';
    eraseBtn.style.backgroundColor = isErasing ? '#e74c3c' : '#1abc9c';
    resetDebounceTimer();
});

// Xóa toàn bộ canvas
deleteBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
    document.getElementById('probabilities').innerHTML = '';
    probChart.data.datasets[0].data = Array(10).fill(0);
    probChart.update();
    // probChart.erase();
    isErasing = false;
    ctx.strokeStyle = 'black';
    eraseBtn.textContent = 'Erase';
    eraseBtn.style.backgroundColor = '#1abc9c';
    resetDebounceTimer();
});

// Khởi tạo biểu đồ xác suất
const probChart = new Chart(document.getElementById('probChart'), {
    type: 'bar',
    data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
            label: 'Xác suất (%)',
            data: Array(10).fill(0),
            backgroundColor: 'rgba(54, 162, 235, 0.6)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            y: { beginAtZero: true, max: 100, title: { display: true, text: 'Xác suất (%)' } },
            x: { title: { display: true, text: 'Chữ số' } }
        },
        plugins: { legend: { display: false } }
    }
});

// // Hàm gửi dự đoán
// function sendPrediction() {
//     const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
//     const isEmpty = Array.from(imgData).every(pixel => pixel === 255);
//     if (isEmpty) {
//         // Không gửi nếu canvas trống
//         document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
//         probChart.data.datasets[0].data = Array(10).fill(0);
//         probChart.update();
//         return;
//     }

//     const dataURL = canvas.toDataURL('image/png');
//     fetch('/predict', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ image: dataURL })
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert(data.error);
//             return;
//         }
//         document.getElementById('predictedDigit').textContent = `Chữ số dự đoán: ${data.digit}`;
//         const probsList = document.getElementById('probabilities');
//         // probsList.innerHTML = data.probabilities.map((p, i) => `<li>Chữ số ${i}: ${(p * 100).toFixed(2)}%</li>`).join('');
//         probChart.data.datasets[0].data = data.probabilities.map(p => p * 100);
//         probChart.update();
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('Lỗi khi gửi yêu cầu dự đoán');
//     });
// }

function sendPrediction() {
    // Lấy dữ liệu gốc
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData;

    // Tìm bounding box
    let minX = width, minY = height, maxX = 0, maxY = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = (y * width + x) * 4;
            const alpha = data[index + 3];
            const isNotWhite = data[index] !== 255 || data[index + 1] !== 255 || data[index + 2] !== 255;

            if (isNotWhite && alpha > 0) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    // Nếu không có nét vẽ -> không dự đoán
    if (minX > maxX || minY > maxY) {
        document.getElementById('predictedDigit').textContent = 'Chưa có dự đoán';
        probChart.data.datasets[0].data = Array(10).fill(0);
        probChart.update();
        return;
    }

    // Cắt ảnh theo bounding box
    const croppedWidth = maxX - minX + 1;
    const croppedHeight = maxY - minY + 1;
    const croppedImageData = ctx.getImageData(minX, minY, croppedWidth, croppedHeight);

    // Tạo canvas tạm 28x28 (hoặc canvas.width x canvas.height)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');

    // Fill trắng
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Tính vị trí để vẽ vào giữa
    const offsetX = Math.floor((canvas.width - croppedWidth) / 2);
    const offsetY = Math.floor((canvas.height - croppedHeight) / 2);

    // Vẽ phần đã cắt vào giữa
    tempCtx.putImageData(croppedImageData, offsetX, offsetY);

    // Lấy dataURL từ canvas đã căn giữa
    const dataURL = tempCanvas.toDataURL('image/png');

    // Gửi ảnh đi như cũ
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        document.getElementById('predictedDigit').textContent = `Chữ số dự đoán: ${data.digit}`;
        probChart.data.datasets[0].data = data.probabilities.map(p => p * 100);
        probChart.update();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Lỗi khi gửi yêu cầu dự đoán');
    });
}


// Hàm reset timer cho dự đoán tự động
function resetDebounceTimer() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(sendPrediction, 500); // Chờ 1 giây
}

// Dự đoán khi nhấn nút Predict
predictBtn.addEventListener('click', sendPrediction);   
deleteBtn.addEventListener('click', () => {
    resetDebounceTimer();
});