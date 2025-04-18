import os
import faiss
import torch
import clip
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, session
from flask_cors import CORS
from flask_session import Session
import pymongo
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import timedelta
import time
import subprocess
from werkzeug.utils import secure_filename

# --- CLIP & FAISS Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_embedding(pil_image):
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(embedding)
    return embedding

# --- Flask & App Configuration ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fastians")
app.permanent_session_lifetime = timedelta(days=30)
CORS(app, supports_credentials=True)

# MongoDB Connection
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://tahamishi12:fastians@cluster0.1fal6.mongodb.net/"
)
client = pymongo.MongoClient(MONGO_URI)
db = client["user_database"]
users_collection = db["users"]

# Flask-Session Configuration
app.config.update({
    "SESSION_TYPE":       "mongodb",
    "SESSION_PERMANENT":  True,
    "SESSION_USE_SIGNER": True,
    "SESSION_MONGODB":    client,
    "SESSION_MONGODB_DB": "user_database",
    "SESSION_MONGODB_COLLECTION": "sessions",
    "SESSION_COOKIE_SAMESITE":   "None",
    "SESSION_COOKIE_SECURE":     True,
})
Session(app)

# External APIs
CHATBOT_URL          = os.environ.get("CHATBOT_URL", "https://chat.aezenai.com")
IMAGE_PREDICTION_API = os.environ.get(
    "IMAGE_PREDICTION_API",
    "https://imagexclassifier-1.onrender.com/api/predict"
)

# Debug Browser Versions
try:
    print("Chromium:", subprocess.check_output(["chromium","--version"]).decode().strip())
    print("ChromeDriver:", subprocess.check_output(["chromedriver","--version"]).decode().strip())
except Exception as e:
    print("Browser version check failed:", e)

# Persist query‑string email into session
@app.before_request
def set_email_from_query():
    if not session.get("user_email"):
        email = request.args.get("email")
        if email:
            session["user_email"] = email

# --- Routes matching your templates ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_route():
    return render_template("train.html")

@app.route("/predict")
def predict_route():
    return render_template("predict.html")

@app.route("/unlearn")
def unlearn_route():
    return render_template("unlearn.html")

@app.route("/check-user", methods=["GET"])
def check_user():
    return jsonify({"email": session.get("user_email")})

@app.route("/register-email", methods=["POST"])
def register_email():
    data = request.json or {}
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email is required"}), 400

    users_collection.insert_one({"email": email})
    session["user_email"] = email
    session.permanent = True
    return jsonify({"message": "Email registered successfully", "success": True})

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Ensure user email in session or form
    email = session.get("user_email")
    if not email:
        new_email = (
            request.form.get("email")
            or (request.get_json() or {}).get("email")
        )
        if not new_email:
            return jsonify({"error": "User email not found"}), 400
        if not users_collection.find_one({"email": new_email}):
            users_collection.insert_one({"email": new_email})
        session["user_email"] = new_email
        session.permanent = True
        email = new_email

    image = request.files["image"]
    filename = secure_filename(image.filename)
    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    image.save(filepath)

    pil_image = Image.open(filepath).convert("RGB")
    embedding = compute_clip_embedding(pil_image)

    try:
        with open(filepath, "rb") as img_file:
            resp = requests.post(
                IMAGE_PREDICTION_API,
                files={"test_image": img_file}
            )
            resp.raise_for_status()
            result = resp.json().get("result")
            if not result:
                raise ValueError("No result from model")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    automate_chatbot(email, result)
    return jsonify({"message": "Image processed and sent to chatbot", "popup": True})

# --- Chatbot Automation ---
def automate_chatbot(email, image_name):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(CHATBOT_URL)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "email"))
        ).send_keys(email)

        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "Button_btn___t8GZ"))
        ).click()

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ant-input"))
        ).send_keys(f"give me buying link of {image_name} and its description")

        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "icon-send"))
        ).click()

        time.sleep(5)
    except Exception as e:
        print("Chatbot automation error:", e)
    finally:
        driver.quit()

# --- Run Server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
