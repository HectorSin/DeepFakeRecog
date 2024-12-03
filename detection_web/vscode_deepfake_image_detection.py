UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# GPU 설정 및 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# 클래스 매핑
idx_to_cls = {0: 'FAKE', 1: 'REAL'}

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

# 단일 이미지 예측
def predict_image(image_path, model):
    face_tensor, face_pil = detect_and_crop_face(image_path)
    if face_tensor is None:
        return None, None
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probabilities, 1)
    predicted_label = idx_to_cls[predicted.item()]
    return predicted_label, face_pil

# 모델 로드 함수
def load_model(model_path, num_classes):
    model = models.efficientnet_v2_s(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 모델 로드
model_path = '/workspace/EfficientNetV2/EfficientNETV2s_imagenet_pretrained_Final_best_model.pth'
model = load_model(model_path, num_classes=2)
