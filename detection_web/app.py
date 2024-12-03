from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import atexit
import signal

app = Flask(__name__)

def shutdown_server():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)

atexit.register(shutdown_server)

# Configurations
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# EfficientNet 모델 설정
idx_to_cls = {0: 'FAKE', 1: 'REAL'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# 모델 로드 함수
def load_model(model_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = r'EfficientNETV2s_imagenet_pretrained_Final_best_model.pth'
model = load_model(model_path, num_classes=2)

# 얼굴 탐지 및 크롭 함수
def detect_and_crop_face(image_path):
    img = Image.open(image_path).convert("RGB")
    faces = mtcnn(img)
    if faces is not None:
        face = faces[0] if faces.ndim == 4 else faces
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_np = face.permute(1, 2, 0).cpu().numpy()
        face_pil = Image.fromarray((face_np * 255).astype(np.uint8))
        face_tensor = transform(face_pil).unsqueeze(0)
        return face_tensor, face_pil
    else:
        return None, None
    
# 단일 이미지 예측 함수
def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")  # 원본 이미지를 그대로 사용
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    predicted_label = idx_to_cls[predicted.item()]
    return predicted_label, None


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/picture_detection', methods=['GET', 'POST'])
def picture_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('picture_detection_1.html', error="파일이 업로드되지 않았습니다.")
        file = request.files['file']
        if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # 원본 파일 저장
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 딥페이크 탐지 수행 (얼굴 크롭 과정 생략)
            predicted_label, _ = predict_image(file_path, model)  # 얼굴 크롭하지 않음
            if predicted_label is None:
                return render_template('picture_detection.html', error="얼굴을 탐지하지 못했습니다.")

            # 텍스트 박스 내용 설정
            auto_text = "#DeepFake" if predicted_label == "FAKE" else ""


            # 결과 페이지 렌더링
            return render_template(
                'picture_detection.html',
                result=f"탐지 결과: {'Deepfake' if predicted_label == 'FAKE' else 'REAL'}",
                result_image=url_for('static', filename=f'uploads/{filename}'),  # 원본 이미지를 표시
                auto_text=auto_text  # 텍스트 박스에 넣을 내용
            )
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