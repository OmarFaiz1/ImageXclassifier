from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_session import Session
import os
import shutil
import time
from datetime import timedelta
import subprocess
import pymongo
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import torch
import clip
from PIL import Image
import numpy as np
import faiss

# ----------------------------- Configuration -----------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fastians")
app.permanent_session_lifetime = timedelta(days=30)

CORS(app, supports_credentials=True)

UPLOAD_FOLDER = 'static/uploads'
TRAIN_FOLDER = 'static/train_data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Setup
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://tahamishi12:fastians@cluster0.1fal6.mongodb.net/")
client = pymongo.MongoClient(MONGO_URI)
db = client["user_database"]
users_collection = db["users"]
session_collection = db["sessions"]
image_collection = db["images"]

# Flask-Session
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_MONGODB"] = client
app.config["SESSION_MONGODB_DB"] = "user_database"
app.config["SESSION_MONGODB_COLLECTION"] = "sessions"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
Session(app)

# Optional external prediction API (fallback)
IMAGE_PREDICTION_API = os.environ.get("IMAGE_PREDICTION_API", "")
CHATBOT_URL = os.environ.get("CHATBOT_URL", "https://your-chatbot-url.com")

# ----------------------------- Model Setup -----------------------------

model, preprocess = clip.load("ViT-B/32", device="cpu")
index = None
image_paths = []

# ----------------------------- Utility Functions -----------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_to_chatbot(image_name, email):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(CHATBOT_URL)
        time.sleep(5)
        input_box = driver.find_element(By.ID, "message_input")
        input_box.send_keys(f"Image: {image_name}, Email: {email}")
        send_button = driver.find_element(By.ID, "send_button")
        send_button.click()
        time.sleep(2)
    except Exception as e:
        print(f"Chatbot automation error: {e}")
    finally:
        driver.quit()

@app.before_request
def set_email_from_query():
    if not session.get("user_email"):
        email = request.args.get("email")
        if email:
            session["user_email"] = email
            print(f"Email set from query parameter: {email}")

# ----------------------------- Routes -----------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register-email', methods=["POST"])
def register_email():
    data = request.json
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email is required"}), 400

    existing_user = users_collection.find_one({"email": email})
    if not existing_user:
        users_collection.insert_one({"email": email})

    session["user_email"] = email
    session.permanent = True

    return jsonify({"message": "Email registered successfully", "success": True})

@app.route('/train', methods=['GET', 'POST'])
def train_route():
    global index, image_paths
    image_paths = []
    image_files = [f for f in os.listdir(TRAIN_FOLDER) if allowed_file(f)]
    image_features = []

    for filename in image_files:
        path = os.path.join(TRAIN_FOLDER, filename)
        image_paths.append(path)
        image = preprocess(Image.open(path)).unsqueeze(0).to("cpu")
        with torch.no_grad():
            features = model.encode_image(image)
            image_features.append(features[0].numpy())

    image_features = np.array(image_features).astype("float32")
    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features)

    return render_template('train.html', message=f"Training completed with {len(image_files)} images.")

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    global index, image_paths

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return render_template('predict.html', message="Invalid file uploaded.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = preprocess(Image.open(filepath)).unsqueeze(0).to("cpu")
        with torch.no_grad():
            query_feature = model.encode_image(image).detach().numpy().astype("float32")

        if index is None or not image_paths:
            return render_template('predict.html', message="No trained data available.")

        D, I = index.search(query_feature, k=1)
        matched_image = os.path.basename(image_paths[I[0][0]])
        matched_name = os.path.splitext(matched_image)[0]

        user_email = session.get('user_email', 'unknown')
        image_collection.insert_one({
            'session': user_email,
            'uploaded_image': filename,
            'matched_image': matched_image,
            'matched_name': matched_name
        })

        send_to_chatbot(matched_name, user_email)

        return render_template('predict.html', matched_name=matched_name, uploaded_image=filename, matched_image=matched_image)

    return render_template('predict.html')

@app.route('/upload-image', methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = request.files["image"]
        email = session.get("user_email")

        if not email:
            return jsonify({"error": "User email not found in session"}), 400

        response = requests.post(IMAGE_PREDICTION_API, files={"test_image": image})
        response.raise_for_status()
        result = response.json().get("result")

        if not result:
            return jsonify({"error": "No result from model"}), 500

        send_to_chatbot(result, email)
        return jsonify({"message": "Image processed and sent to chatbot", "popup": True})
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/unlearn', methods=['GET', 'POST'])
def unlearn_route():
    if request.method == 'POST':
        session.clear()
        return redirect(url_for('home'))
    return render_template('unlearn.html')

@app.route('/check-user', methods=["GET"])
def check_user():
    email = session.get("user_email")
    return jsonify({"email": email}) if email else jsonify({"email": None})

# ----------------------------- Start App -----------------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
