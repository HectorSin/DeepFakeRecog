import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
import os
from flask import Flask, request, jsonify, send_from_directory

# Flask 설정
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 클래스 매핑
idx_to_cls = {0: 'fake', 1: 'real'}

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# 워터마크 탐지 함수 정의
def detect_watermark(input_image_path, watermark_image_path, start_x=50, start_y=50):
    img = cv2.imread(input_image_path)
    watermark = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"입력 이미지 파일을 찾을 수 없습니다: {input_image_path}")
    if watermark is None:
        raise FileNotFoundError(f"워터마크 이미지 파일을 찾을 수 없습니다: {watermark_image_path}")

    _, binary_watermark = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY_INV)
    wm_height, wm_width = binary_watermark.shape

    for i in range(wm_height):
        for j in range(wm_width):
            r_value = img[start_y + i, start_x + j, 2]
            lsb = r_value & 1
            if lsb != binary_watermark[i, j]:
                return False  # 워터마크 없음

    return True  # 워터마크 있음

# 모델 로드 함수
def load_model(model_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

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

# 통합 로직
def main(image_path, model_path, watermark_image_path):
    model = load_model(model_path, num_classes=2)

    # Step 1: 얼굴 탐지 및 모델 예측
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

    # Step 2: 워터마크 탐지
    watermark_detected = detect_watermark(image_path, watermark_image_path)
    if watermark_detected:
        return {'status': 'success', 'result': 'fake'}
    else:
        return {'status': 'success', 'result': 'real'}

# API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '파일이 없습니다.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '파일 이름이 비어 있습니다.'})

    if file and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 모델 경로와 워터마크 경로
        model_path = 'static/models/EfficientNETV2s_imagenet_pretrained_Final_best_model.pth'
        watermark_path = 'static/watermarks/fakeme_wm.png'

        result = main(filepath, model_path, watermark_path)
        return jsonify(result)

    return jsonify({'status': 'error', 'message': '허용되지 않는 파일 형식입니다.'})

# 실행
if __name__ == '__main__':
    app.run(debug=True)
