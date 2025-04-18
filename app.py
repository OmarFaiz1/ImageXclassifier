from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import pymongo
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import timedelta
import time
import subprocess
import faiss
import torch
import clip
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fastians")
app.permanent_session_lifetime = timedelta(days=30)

# Enable CORS with credentials
CORS(app, supports_credentials=True)

# MongoDB Connection
MONGO_URI = "mongodb+srv://tahamishi12:fastians@cluster0.1fal6.mongodb.net/"
if not MONGO_URI:
    raise ValueError("MongoDB URI is missing. Please set MONGO_URI in your environment variables.")
print("Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI)
db = client["user_database"]
users_collection = db["users"]
print("Connected to MongoDB successfully!")

# Configure Flask-Session with MongoDB
app.config["SESSION_TYPE"] = "mongodb"
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_MONGODB"] = client
app.config["SESSION_MONGODB_DB"] = "user_database"
app.config["SESSION_MONGODB_COLLECTION"] = "sessions"
Session(app)
app.config["SESSION_COOKIE_SAMESITE"] = "None"  # Allow cross-site usage in iframes
app.config["SESSION_COOKIE_SECURE"] = True       # Required when SameSite=None (HTTPS only)

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Environment variables
CHATBOT_URL = os.environ.get("CHATBOT_URL", "https://chat.aezenai.com")
IMAGE_PREDICTION_API = os.environ.get("IMAGE_PREDICTION_API", "https://imagexclassifier-1.onrender.com/api/predict")

# Set up device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Global data holders for image similarity
image_paths = []  # Stores image file paths
image_embeddings = []  # Stores corresponding embeddings
index = None  # FAISS index will be initialized later

# Debugging: Print Chromium and ChromeDriver versions
try:
    chromium_version = subprocess.check_output(["chromium", "--version"]).decode().strip()
    chromedriver_version = subprocess.check_output(["chromedriver", "--version"]).decode().strip()
    print(f"Chromium Version: {chromium_version}")
    print(f"ChromeDriver Version: {chromedriver_version}")
except Exception as e:
    print(f"Error fetching browser versions: {str(e)}")

# --- Helper Functions for Image Similarity ---

def compute_embedding(pil_image):
    """
    Compute CLIP embedding for a given PIL image and normalize it.
    """
    if model is None or preprocess is None:
        raise RuntimeError("CLIP model not initialized.")
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(embedding)  # Normalize for cosine similarity
    return embedding

def initialize_faiss_index(embedding_dim):
    """
    Initializes an HNSW FAISS index optimized for cosine similarity.
    """
    hnsw_index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M = 32 for better recall
    hnsw_index.hnsw.efSearch = 64  # Higher means better recall, slower query
    return hnsw_index

def build_index():
    """
    Builds the FAISS index from stored image embeddings.
    """
    global index
    if not image_embeddings:
        return
    embeddings_matrix = np.vstack(image_embeddings).astype(np.float32)
    index = initialize_faiss_index(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

def find_similar_images(query_embedding, k=5):
    """
    Finds top k similar images to the query embedding.
    """
    if index is None or index.ntotal == 0:
        return [], []
    distances, indices = index.search(query_embedding, k)
    results = [image_paths[i] for i in indices[0]]
    return results, distances[0]

# --- Flask Routes ---

@app.before_request
def set_email_from_query():
    if not session.get("user_email"):
        email = request.args.get("email")
        if email:
            session["user_email"] = email
            print(f"Email set from query parameter: {email}")

@app.route("/")
def index():
    print("Serving index.html")
    return render_template("index.html")

@app.route("/check-user", methods=["GET"])
def check_user():
    email = session.get("user_email")
    print(f"Checking user session: {email}")
    return jsonify({"email": email}) if email else jsonify({"email": None})

@app.route("/register-email", methods=["POST"])
def register_email():
    try:
        data = request.json
        email = data.get("email")
        print(f"Received email for registration: {email}")
        if not email:
            print("Error: Email is missing.")
            return jsonify({"error": "Email is required"}), 400
        users_collection.insert_one({"email": email})
        print(f"Email {email} inserted into MongoDB.")
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
            print("Error: No image provided.")
            return jsonify({"error": "No image provided"}), 400

        email = session.get("user_email")
        if not email:
            new_email = request.form.get("email") or request.args.get("email")
            if not new_email and request.is_json:
                data = request.get_json()
                if data:
                    new_email = data.get("email")
            if new_email:
                existing_user = users_collection.find_one({"email": new_email})
                if not existing_user:
                    users_collection.insert_one({"email": new_email})
                session["user_email"] = new_email
                session.permanent = True
                email = new_email
                print(f"Email {new_email} found in request and stored in session.")
            else:
                print("Error: User email not found in session or request.")
                return jsonify({"error": "User email not found"}), 400

        image = request.files["image"]
        print副书记("Uploading image for email: {email}")

        response = requests.post(IMAGE_PREDICTION_API, files={"test_image": image})
        response.raise_for_status()
        result = response.json().get("result")
        if not result:
            print("Error: No result received from model.")
            return jsonify({"error": "No result from model"}), 500

        print(f"Image processed successfully. Result: {result}")
        automate_chatbot(email, result)

        return jsonify({"message": "Image processed and sent to chatbot", "popup": True})
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return jsonify({"error": f"API request failed: {str(e)}"}), 500
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/add-image", methods=["POST"])
def add_image():
    """
    Handles new image uploads to add to the dataset.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        pil_image = Image.open(file_path).convert("RGB")
        embedding = compute_embedding(pil_image)
        image_paths.append(file_path)
        image_embeddings.append(embedding)
        build_index()
        return jsonify({"message": f"Image '{filename}' added successfully!"})
    return jsonify({"error": "Error uploading image"}), 500

@app.route("/find-similar", methods=["POST"])
def find_similar():
    """
    Handles image upload and returns similar images from the dataset.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        pil_image = Image.open(file_path).convert("RGB")
        query_embedding = compute_embedding(pil_image)
        if index is None or index.ntotal == 0:
            return jsonify({"error": "No images in the dataset"}), 404
        similar_image_paths, distances = find_similar_images(query_embedding)
        return jsonify({"similar_images": similar_image_paths, "distances": distances.tolist()})
    return jsonify({"error": "Error uploading image"}), 500

def automate_chatbot(email, image_name):
    """
    Automates interaction with the chatbot using Selenium WebDriver.
    """
    print(f"Starting chatbot automation for email: {email}, image: {image_name}")
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    try:
        print(f"Opening chatbot URL: {CHATBOT_URL}")
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

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
