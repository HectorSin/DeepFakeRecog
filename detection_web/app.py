from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2

# Flask 설정
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MODEL_PATH'] = 'static/models/EfficientNETV2s_imagenet_pretrained_Final_best_model.pth'
app.config['WATERMARK_PATH'] = 'static/watermarks/fakeme_wm.png'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
idx_to_cls = {0: 'fake', 1: 'real'}

# 모델 로드 함수
def load_model(model_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 워터마크 탐지 함수
def detect_watermark(input_image_path, watermark_image_path, start_x=50, start_y=50):
    img = cv2.imread(input_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    if img is None or watermark is None:
        return False

    _, binary_watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY_INV)
    wm_height, wm_width = binary_watermark.shape

    for i in range(wm_height):
        for j in range(wm_width):
            r_value = img[start_y + i, start_x + j, 2]
            lsb = r_value & 1
            if lsb != binary_watermark[i, j]:
                return False  # 워터마크가 없음
    return True  # 워터마크가 있음

# 얼굴 탐지 및 크롭 함수
def detect_and_crop_face(image_path, normalize=True):
    img = Image.open(image_path).convert("RGB")
    faces = mtcnn(img)

    if faces is not None:
        face = faces[0] if faces.ndim == 4 else faces
        transform_steps = [transforms.Resize((224, 224))]
        if normalize:
            transform_steps += [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            transform_steps += [transforms.ToTensor()]

        transform = transforms.Compose(transform_steps)
        face_np = face.permute(1, 2, 0).cpu().numpy()
        face_pil = Image.fromarray((face_np * 255).astype(np.uint8))
        face_tensor = transform(face_pil).unsqueeze(0)
        return face_tensor, face_pil
    else:
        return None, None

# 모델 예측 및 워터마크 탐지
def predict_and_detect(image_path, model_path, watermark_image_path):
    model = load_model(model_path, num_classes=2)
    face_tensor, _ = detect_and_crop_face(image_path, normalize=True)
    if face_tensor is None:
        return {'status': 'error', 'message': '얼굴 탐지 실패'}

    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)

    predicted_label = idx_to_cls[predicted.item()]
    if predicted_label == 'fake':
        return {'status': 'success', 'result': 'fake'}

    watermark_detected = detect_watermark(image_path, watermark_image_path)
    return {'status': 'success', 'result': 'fake' if watermark_detected else 'real'}

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/picture_detection', methods=['GET', 'POST'])
def picture_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('picture_detection.html', error="파일이 업로드되지 않았습니다.")
        file = request.files['file']
        if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            model_path = app.config['MODEL_PATH']
            watermark_path = app.config['WATERMARK_PATH']
            result = predict_and_detect(file_path, model_path, watermark_path)

            if result['status'] == 'success':
                auto_text = "#DeepFake" if result['result'] == 'fake' else ""
                return render_template(
                    'picture_detection.html',
                    result=f"탐지 결과: {'Deepfake' if result['result'] == 'fake' else 'REAL'}",
                    result_image=url_for('static', filename=f'uploads/{filename}'),
                    auto_text=auto_text
                )
            return render_template('picture_detection.html', error=result['message'])
        return render_template('picture_detection.html', error="유효하지 않은 파일 형식입니다.")
    return render_template('picture_detection.html', auto_text="")

@app.route('/picture_generation')
def picture_generation():
    return render_template('picture_generation.html')

@app.route('/picture_generation2')
def picture_generation2():
    return render_template('picture_generation2.html')

@app.route('/video_detection')
def video_detection():
    return render_template('video_detection.html')

@app.route('/video_generation')
def video_generation():
    return render_template('video_generation.html')

@app.route('/video_generation2')
def video_generation2():
    return render_template('video_generation2.html')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
