from flask import Flask, request, render_template, send_from_directory
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
import os

# Flask 애플리케이션 초기화
app = Flask(__name__)

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

UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

opt = TestOptions().parse()
opt.crop_size = 224
opt.name = "people"
opt.Arc_path = "arcface_model/arcface_checkpoint.tar"
model = create_model(opt)
model.eval()

def process_image(pic_a_path, pic_b_path, output_path):
    with torch.no_grad():
        img_a = Image.open(pic_a_path).convert("RGB")
        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).cuda()

        img_b = Image.open(pic_b_path).convert("RGB")
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2]).cuda()

        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to("cpu")
        latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
        latend_id = latend_id.to("cuda")

        img_fake = model(img_id, img_att, latend_id, latend_id, True)
        img_fake = img_fake[0].detach().cpu().numpy()
        img_fake = np.transpose(img_fake, (1, 2, 0)) * 255
        img_fake = cv2.cvtColor(img_fake.astype("uint8"), cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, img_fake)

@app.route("/", methods=["GET", "POST"])
def index():
    result_image_path = None
    if request.method == "POST":
        pic_a = request.files["pic_a"]
        pic_b = request.files["pic_b"]

        pic_a_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_a.jpg")
        pic_b_path = os.path.join(app.config["UPLOAD_FOLDER"], "pic_b.jpg")
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], "result.jpg")

        pic_a.save(pic_a_path)
        pic_b.save(pic_b_path)

        process_image(pic_a_path, pic_b_path, output_path)

        result_image_path = "/output/result.jpg"

    return render_template("index.html", result_image=result_image_path)

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
