# -*- coding: utf-8 -*-
"""vscode_Deepfake_video_detection

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JN6JWrYN9JsJOg9OExenkFKEytwXME70
"""

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
import os

# GPU/CPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTCNN 모델 초기화
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)  # 단일 얼굴 감지에 사용

# 이미지 정규화 복원 함수
def denormalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    return image

# 모델 로드 함수
def load_model(model_path, num_classes):
    model = torch.hub.load('pytorch/vision', 'efficientnet_v2_s', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model

# 얼굴 감지 및 전처리 함수
def detect_face_and_preprocess(frame, transform):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    face = mtcnn(frame_pil)  # 얼굴 감지 및 크롭
    if face is None:
        return None

    face_tensor = face.unsqueeze(0)  # 배치 차원 추가
    return face_tensor

# 영상에서 프레임 추출 및 Deepfake 판별
def process_video(video_path, model, idx_to_cls, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        print("Error: Unable to retrieve FPS from video.")
        return
    frame_interval = fps // frame_rate if fps > 0 else 1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    fake_count = 0
    real_count = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        if total_frames % frame_interval != 0:
            continue

        # 얼굴 감지 및 전처리
        face_tensor = detect_face_and_preprocess(frame, transform)
        if face_tensor is None:
            continue

        # 모델 예측
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            if predicted.item() == 0:
                fake_count += 1
            else:
                real_count += 1

    cap.release()

    print(f"Total Frames Processed: {fake_count + real_count}")
    print(f"Fake Frames: {fake_count}, Real Frames: {real_count}")

    if fake_count > real_count:
        print("Video is classified as DEEPFAKE.")
        return "DEEPFAKE"
    else:
        print("Video is classified as REAL.")
        return "REAL"

# 클래스 라벨 딕셔너리 생성
idx_to_cls = {0: 'FAKE', 1: 'REAL'}

# 모델 경로 및 영상 경로 설정
model_path = '/workspace/EfficientNetV2/EfficientNETV2s_imagenet_pretrained_Final_best_model.pth'
video_path = '/workspace/EfficientNetV2/suzy_deepfake.mp4'

# 모델 로드
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    model = load_model(model_path, num_classes=2)

# Deepfake 판별
if os.path.exists(video_path):
    process_video(video_path, model, idx_to_cls)
else:
    print(f"Error: Video file not found at {video_path}")