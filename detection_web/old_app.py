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
from flask import Flask, request, render_template, send_from_directory, url_for

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
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# 모델 초기화
opt = TestOptions().parse()
opt.crop_size = 224
opt.name = "people"
opt.Arc_path = "arcface_model/arcface_checkpoint.tar"
model = create_model(opt)
model.eval()

# GPU가 없는 환경에서도 작동 가능하도록 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def process_image(pic_a_path, pic_b_path, output_path):
    with torch.no_grad():
        # 이미지 A 처리
        img_a = Image.open(pic_a_path).convert("RGB")
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(device)

        # 이미지 B 처리
        img_b = Image.open(pic_b_path).convert("RGB")
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).to(device)

        # Arcface 계산
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to("cpu")
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to(device)

        # 이미지 생성
        img_fake = model(img_id, img_att, latend_id, latend_id, True)
        img_fake = img_fake[0].detach().cpu().numpy()
        img_fake = np.transpose(img_fake, (1, 2, 0)) * 255
        img_fake = cv2.cvtColor(img_fake.astype("uint8"), cv2.COLOR_RGB2BGR)

        # 결과 저장
        cv2.imwrite(output_path, img_fake)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image_path = None
    if request.method == "POST":
        # 업로드된 파일 가져오기
        pic_a = request.files.get("pic_a")
        pic_b = request.files.get("pic_b")

        if pic_a and pic_b:
            pic_a_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_a.jpg")
            pic_b_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_b.jpg")
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], "result.jpg")

            # 저장 후 처리
            pic_a.save(pic_a_path)
            pic_b.save(pic_b_path)
            process_image(pic_a_path, pic_b_path, output_path)

            # 결과 이미지 경로
            result_image_path = url_for("output_file", filename="result.jpg")

    return render_template("picture_generation.html", result_image=result_image_path)

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
