import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
import os
import atexit
import signal
from flask import Flask, request, render_template, send_from_directory, url_for, make_response

app = Flask(__name__)

# 서버 종료 핸들러
def shutdown_server():
    pid = os.getpid()
    os.kill(pid, signal.SIGINT)

atexit.register(shutdown_server)

# Transformer 초기화
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transformer_Arcface = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 업로드/출력 폴더 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.before_first_request
def initialize_model():
    global model
    if model is None:
        opt = TestOptions().parse()
        opt.crop_size = 224
        opt.name = "people"
        opt.Arc_path = "arcface_model/arcface_checkpoint.tar"

        model = create_model(opt)
        model.eval()
        model.to(device)

def process_image(pic_a_path, pic_b_path, output_path):
    with torch.no_grad():
        img_a = Image.open(pic_a_path).convert("RGB")
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(device)

        img_b = Image.open(pic_b_path).convert("RGB")
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).to(device)

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to("cpu")
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(device)

        img_fake = model(img_id, img_att, latend_id, latend_id, True)
        img_fake = img_fake[0].detach().cpu().numpy()
        img_fake = np.transpose(img_fake, (1, 2, 0)) * 255
        img_fake = cv2.cvtColor(img_fake.astype("uint8"), cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img_fake)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image_path = None
    if request.method == "POST":
        pic_a = request.files.get("pic_a")
        pic_b = request.files.get("pic_b")

        if pic_a and pic_b:
            pic_a_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_a.jpg")
            pic_b_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_b.jpg")
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], "result.jpg")

            pic_a.save(pic_a_path)
            pic_b.save(pic_b_path)
            process_image(pic_a_path, pic_b_path, output_path)

            result_image_path = url_for("output_file", filename="result.jpg")

    # 기본 응답 생성
    response = make_response(render_template("picture_generation.html", result_image=result_image_path))
    
    # Keep-Alive 헤더 추가
    response.headers["Connection"] = "keep-alive"
    response.headers["Keep-Alive"] = "timeout=5, max=1000"  # 예: 5초 타임아웃, 최대 1000개의 연결 유지
    
    return response

@app.route("/home", methods=["GET", "POST"])
def home():
    result_image_path = None
    if request.method == "POST":
        pic_a = request.files.get("pic_a")
        pic_b = request.files.get("pic_b")

        if pic_a and pic_b:
            pic_a_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_a.jpg")
            pic_b_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_b.jpg")
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], "result.jpg")

            pic_a.save(pic_a_path)
            pic_b.save(pic_b_path)
            process_image(pic_a_path, pic_b_path, output_path)

            result_image_path = url_for("output_file", filename="result.jpg")

    # 기본 응답 생성
    response = make_response(render_template("picture_generation.html", result_image=result_image_path))
    
    # Keep-Alive 헤더 추가
    response.headers["Connection"] = "keep-alive"
    response.headers["Keep-Alive"] = "timeout=5, max=1000"  # 예: 5초 타임아웃, 최대 1000개의 연결 유지
    
    return response

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
