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
    """
    Compute CLIP embedding for a given PIL image and normalize it.
    """
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
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://tahamishi12:fastians@cluster0.1fal6.mongodb.net/")
if not MONGO_URI:
    raise ValueError("MongoDB URI is missing. Please set MONGO_URI in your environment variables.")

client = pymongo.MongoClient(MONGO_URI)
db = client["user_database"]
users_collection = db["users"]

# Flask-Session Configuration
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_MONGODB"] = client
app.config["SESSION_MONGODB_DB"] = "user_database"
app.config["SESSION_MONGODB_COLLECTION"] = "sessions"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True
Session(app)

# Environment variables
CHATBOT_URL = os.environ.get("CHATBOT_URL", "https://chat.aezenai.com")
IMAGE_PREDICTION_API = os.environ.get("IMAGE_PREDICTION_API", "https://imagexclassifier-1.onrender.com/api/predict")

# Debug Browser Versions
try:
    chromium_version = subprocess.check_output(["chromium", "--version"]).decode().strip()
    chromedriver_version = subprocess.check_output(["chromedriver", "--version"]).decode().strip()
    print(f"Chromium Version: {chromium_version}")
    print(f"ChromeDriver Version: {chromedriver_version}")
except Exception as e:
    print(f"Error fetching browser versions: {str(e)}")

# --- Session Helpers ---
@app.before_request
def set_email_from_query():
    if not session.get("user_email"):
        email = request.args.get("email")
        if email:
            session["user_email"] = email
            print(f"Email set from query parameter: {email}")

# --- Routes ---
@app.route("/")
def index():
    print("Serving index.html")
    return render_template("index.html")

@app.route("/check-user", methods=["GET"])
def check_user():
    email = session.get("user_email")
    print(f"Checking user session: {email}")
    return jsonify({"email": email})

@app.route("/register-email", methods=["POST"])
def register_email():
    try:
        data = request.json
        email = data.get("email")
        print(f"Received email for registration: {email}")
        if not email:
            return jsonify({"error": "Email is required"}), 400
        users_collection.insert_one({"email": email})
        session["user_email"] = email
        session.permanent = True
        print(f"Session created for email: {email}")
        return jsonify({"message": "Email registered successfully", "success": True})
    except Exception as e:
        print(f"Error registering email: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload-image", methods=["POST"])
def upload_image():
    try:
        print("Received image upload request.")
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        # Ensure user email in session or form
        email = session.get("user_email")
        if not email:
            new_email = request.form.get("email") or request.args.get("email")
            if not new_email and request.is_json:
                data = request.get_json()
                new_email = data.get("email") if data else None
            if new_email:
                if not users_collection.find_one({"email": new_email}):
                    users_collection.insert_one({"email": new_email})
                session["user_email"] = new_email
                session.permanent = True
                email = new_email
            else:
                return jsonify({"error": "User email not found"}), 400

        image = request.files["image"]
        print(f"Uploading image for email: {email}")

        # Save locally and compute embedding
        filename = secure_filename(image.filename)
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        image.save(filepath)
        pil_image = Image.open(filepath).convert("RGB")
        embedding = compute_clip_embedding(pil_image)

        # Optionally use external API
        with open(filepath, "rb") as img_file:
            response = requests.post(
                IMAGE_PREDICTION_API,
                files={"test_image": img_file}
            )
        response.raise_for_status()
        result = response.json().get("result")
        if not result:
            return jsonify({"error": "No result from model"}), 500

        automate_chatbot(email, result)
        return jsonify({"message": "Image processed and sent to chatbot", "popup": True})

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

# --- Chatbot Automation ---
def automate_chatbot(email, image_name):
    print(f"Starting chatbot automation for email: {email}, image: {image_name}")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(CHATBOT_URL)
        email_field = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "email"))
        )
        email_field.send_keys(email)

        chat_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "Button_btn___t8GZ"))
        )
        chat_button.click()

        message_box = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ant-input"))
        )
        message_box.send_keys(f"give me buying link of {image_name} and its description")

        send_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "icon-send"))
        )
        send_button.click()

        time.sleep(5)
    except Exception as e:
        print(f"Error in chatbot automation: {str(e)}")
    finally:
        driver.quit()
        print("Browser closed successfully.")

# --- Run Server ---
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
