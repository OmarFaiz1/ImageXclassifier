import os
import io
import pickle
import numpy as np
import torch
import faiss
import clip
from PIL import Image
from flask import Flask, request, redirect, url_for, flash, render_template, jsonify
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from queue import Queue
from waitress import serve
from flask_cors import cross_origin

app = Flask(__name__)
CORS(app, origins=["*"])

# Alternatively, to enable CORS for only specific routes:
# from flask_cors import cross_origin
# @app.route('/api/predict', methods=['POST'])
# @cross_origin()  # Enable CORS for just this route

# Set environment variable to avoid OpenMP duplicate runtime warnings.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------
# File Names for Persistence
# ---------------------------
INDEX_FILE = "faiss_index.index"
PRODUCT_NAMES_FILE = "product_names.pkl"

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "replace_this_with_a_random_secret_key"  # Change for production

# ---------------------------
# Global Variables & Model Setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
embedding_dim = 512  # Embedding dimension for ViT-B/32

# Create a FAISS index for fast similarity search.
index = faiss.IndexFlatL2(embedding_dim)
product_names = []  # List of labels corresponding to training images.

# ---------------------------
# ThreadPoolExecutor for parallel image processing
# ---------------------------
MAX_WORKERS = 50  # Max workers for parallel users
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # Set the limit to 50

# Queue for rate-limiting (this is optional, depending on your needs)
request_queue = Queue(maxsize=MAX_WORKERS)  # Set max concurrent users

# ---------------------------
# Persistence Functions
# ---------------------------
def save_training_data():
    """Save the FAISS index and product_names list to disk."""
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(PRODUCT_NAMES_FILE, 'wb') as f:
            pickle.dump(product_names, f)
        app.logger.info("Training data saved successfully.")
    except Exception as e:
        app.logger.error(f"Error saving training data: {e}")

def load_training_data():
    """Load the FAISS index and product_names list from disk, if they exist."""
    global index, product_names
    if os.path.exists(INDEX_FILE) and os.path.exists(PRODUCT_NAMES_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(PRODUCT_NAMES_FILE, 'rb') as f:
                product_names = pickle.load(f)
            app.logger.info("Training data loaded successfully.")
        except Exception as e:
            app.logger.error(f"Error loading training data: {e}")
    else:
        app.logger.info("No previous training data found. Starting fresh.")

# ---------------------------
# Helper Function to Compute an Image's Embedding
# ---------------------------
def compute_embedding(pil_image):
    """
    Given a PIL image, preprocess it and compute its CLIP embedding.
    Returns a normalized numpy array of shape (1, embedding_dim).
    """
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / norm

# ---------------------------
# Image Processing for Prediction in a Separate Thread
# ---------------------------
def process_image(file):
    image_bytes = file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    user_embedding = compute_embedding(pil_image)
    if index.ntotal == 0:
        return "No training data available. Please train the model first!"
    distances, indices = index.search(user_embedding, 1)
    best_distance_sq = distances[0][0]
    # For normalized embeddings, cosine similarity = 1 - (squared_distance/2)
    confidence = 1 - (best_distance_sq / 2)
    if confidence < 0.8:
        return "Sorry, can't recognize the image. Can you please provide the name instead?"
    match_index = indices.flatten()[0]
    return product_names[match_index]

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    """Home page with navigation links."""
    return render_template("index.html")

# -------- Training Route --------
@app.route('/train', methods=['GET', 'POST'])
def train_route():
    """
    Training page offers:
      - Single Label Training (enter a label and upload one or more images)
      - Bulk Training with two modes:
          • "single" bulk mode: all files come from one folder (the folder name is used as the label)
          • "subfolders" mode: each file’s parent folder is used as the label
    """
    global index, product_names
    if request.method == 'POST':
        train_mode = request.form.get("train_mode")
        if train_mode == "single":
            cloth_name = request.form.get('cloth_name')
            if not cloth_name:
                flash("Please enter a cloth name!")
                return redirect(url_for('train_route'))
            
            files = request.files.getlist("train_images")
            if not files or files[0].filename == "":
                flash("Please upload at least one image!")
                return redirect(url_for('train_route'))
            
            embeddings_list = []
            count = 0
            for file in files:
                try:
                    image_bytes = file.read()
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    embedding = compute_embedding(pil_image)
                    embeddings_list.append(embedding)
                    product_names.append(cloth_name)
                    count += 1
                except Exception as e:
                    flash(f"Error processing image {file.filename}: {e}")
            
            if embeddings_list:
                batch_embeddings = np.vstack(embeddings_list)
                index.add(batch_embeddings)
                save_training_data()
                flash(f"Training complete: {count} image(s) processed for '{cloth_name}'.")
            else:
                flash("No valid images processed.")
            return redirect(url_for('train_route'))
        
        elif train_mode == "bulk":
            bulk_mode = request.form.get("bulk_mode")  # Expected: "subfolders" or "single"
            files = request.files.getlist("bulk_files")
            if not files or files[0].filename == "":
                flash("Please upload a folder with images!")
                return redirect(url_for('train_route'))
            
            count = 0
            if bulk_mode == "single":
                # Expect all files come from one folder; extract folder name automatically.
                folder_names = set()
                for file in files:
                    if "/" in file.filename:
                        folder_names.add(file.filename.split("/")[0])
                if len(folder_names) == 1:
                    auto_label = list(folder_names)[0]
                else:
                    flash("Error: For single folder mode, all files must come from one folder.")
                    return redirect(url_for('train_route'))
                for file in files:
                    try:
                        image_bytes = file.read()
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        embedding = compute_embedding(pil_image)
                        index.add(embedding)
                        product_names.append(auto_label)
                        count += 1
                    except Exception as e:
                        flash(f"Error processing image {file.filename}: {e}")
            elif bulk_mode == "subfolders":
                for file in files:
                    rel_path = file.filename
                    label = None
                    if "/" in rel_path:
                        parts = rel_path.split("/")
                        if len(parts) >= 2 and parts[-2]:
                            label = parts[-2]
                    if not label:
                        flash(f"Skipping {file.filename}: No label could be determined.")
                        continue
                    try:
                        image_bytes = file.read()
                        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        embedding = compute_embedding(pil_image)
                        index.add(embedding)
                        product_names.append(label)
                        count += 1
                    except Exception as e:
                        flash(f"Error processing image {file.filename}: {e}")
            else:
                flash("Invalid bulk training mode selected.")
                return redirect(url_for('train_route'))
            
            if count > 0:
                save_training_data()
                flash(f"Bulk training complete: Processed {count} images.")
            else:
                flash("No valid images processed in bulk training.")
            return redirect(url_for('train_route'))
        
        else:
            flash("Invalid training mode selected.")
            return redirect(url_for('train_route'))
    
    return render_template("train.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    result = None
    if request.method == 'POST':
        file = request.files.get("test_image")
        if not file or file.filename == "":
            flash("Please upload an image for prediction!")
            return redirect(url_for('predict_route'))
        
        # Process image in a separate thread
        future = executor.submit(process_image, file)
        try:
            result = future.result(timeout=30)  # Allow up to 30 seconds for processing
        except TimeoutError:
            flash("The server is busy, please try again later.")
            return redirect(url_for('predict_route'))
    return render_template("predict.html", result=result)

# -------- New API Prediction Route (for client module) --------
@app.route('/api/predict', methods=['POST'])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization'])  # Enable CORS for this route
def api_predict():
    if 'test_image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files.get("test_image")
    if not file or file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    future = executor.submit(process_image, file)
    try:
        result = future.result(timeout=30)
    except TimeoutError:
        return jsonify({"error": "The server is busy, please try again later."}), 500

    return jsonify({"result": result})


# -------- Unlearn Route --------
@app.route('/unlearn', methods=['GET', 'POST'])
def unlearn_route():
    """
    Unlearn page:
      - The user enters a label to remove.
      - All embeddings for that label are removed and the FAISS index is rebuilt.
    """
    global index, product_names
    if request.method == 'POST':
        label_to_remove = request.form.get("unlearn_label")
        if not label_to_remove:
            flash("Please provide a label name to unlearn.")
            return redirect(url_for('unlearn_route'))
        total = index.ntotal
        if total == 0:
            flash("No training data available to unlearn.")
            return redirect(url_for('unlearn_route'))
        new_embeddings = []
        new_labels = []
        removed_count = 0
        for i in range(total):
            if product_names[i] == label_to_remove:
                removed_count += 1
            else:
                vec = index.reconstruct(i)
                new_embeddings.append(vec)
                new_labels.append(product_names[i])
        if removed_count == 0:
            flash(f"No data found for label '{label_to_remove}'.")
            return redirect(url_for('unlearn_route'))
        new_index = faiss.IndexFlatL2(embedding_dim)
        if new_embeddings:
            batch_embeddings = np.vstack(new_embeddings)
            new_index.add(batch_embeddings)
        index = new_index
        product_names = new_labels
        save_training_data()
        flash(f"Successfully unlearned {removed_count} image(s) for label '{label_to_remove}'.")
        return redirect(url_for('home'))
    return render_template("unlearn.html")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    load_training_data()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
