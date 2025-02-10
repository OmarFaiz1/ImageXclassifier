import os
import io
import pickle
import numpy as np
import torch
import faiss
import clip
from PIL import Image
from flask import Flask, request, redirect, url_for, flash, render_template_string

# Optionally, set this environment variable to avoid the OpenMP duplicate runtime warning.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------
# File Names for Persistence
# ---------------------------
INDEX_FILE = "faiss_index.index"
PRODUCT_NAMES_FILE = "product_names.pkl"

# ---------------------------
# Flask App Configuration
# ---------------------------
app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret_key"  # Change this in production

# ---------------------------
# Global Variables & Model Setup
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
embedding_dim = 512  # Embedding dimension for ViT-B/32

# Create a FAISS index for fast similarity search.
# We'll try to load the persisted index; if not available, we initialize a new one.
index = faiss.IndexFlatL2(embedding_dim)
product_names = []  # List to store cloth names (labels) corresponding to each training image.

# ---------------------------
# Persistence Functions
# ---------------------------
def save_training_data():
    """
    Save the FAISS index and product_names list to disk.
    """
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(PRODUCT_NAMES_FILE, 'wb') as f:
            pickle.dump(product_names, f)
        print("Training data saved successfully.")
    except Exception as e:
        print(f"Error saving training data: {e}")

def load_training_data():
    """
    Load the FAISS index and product_names list from disk, if they exist.
    """
    global index, product_names
    if os.path.exists(INDEX_FILE) and os.path.exists(PRODUCT_NAMES_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(PRODUCT_NAMES_FILE, 'rb') as f:
                product_names = pickle.load(f)
            print("Training data loaded successfully.")
        except Exception as e:
            print(f"Error loading training data: {e}")
    else:
        print("No previous training data found. Starting fresh.")

# ---------------------------
# Helper Function to Compute an Image's Embedding
# ---------------------------
def compute_embedding(pil_image):
    """
    Given a PIL image, preprocess it and compute its CLIP embedding.
    Returns a numpy array of shape (1, embedding_dim).
    """
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy().astype(np.float32)
    return embedding

# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    """Home page with links to Train and Predict pages."""
    return render_template_string('''
    <h1>Real-Time Clothing Classifier</h1>
    <ul>
      <li><a href="/train">Train Model</a></li>
      <li><a href="/predict">Predict Clothing</a></li>
    </ul>
    ''')

# -------- Training Route --------
@app.route('/train', methods=['GET', 'POST'])
def train():
    """
    Training page:
      - Users enter a cloth name and upload one or more training images.
      - On submission, a JavaScript function hides the form and shows a loading message.
      - The server processes each image (computing its embedding) and adds it to the FAISS index.
      - The FAISS index and label list are then saved to disk.
    """
    if request.method == 'POST':
        cloth_name = request.form.get('cloth_name')
        if not cloth_name:
            flash("Please enter a cloth name!")
            return redirect(url_for('train'))
        
        files = request.files.getlist("train_images")
        if not files or files[0].filename == "":
            flash("Please upload at least one image!")
            return redirect(url_for('train'))
        
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
            save_training_data()  # Save the updated index and product names.
            flash(f"Training complete: {count} image(s) processed for cloth '{cloth_name}'. Now you can test the model!")
        else:
            flash("No valid images were processed.")
        
        return redirect(url_for('train'))
    
    # GET: Render the training form with a JavaScript-based loading message.
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
  <title>Train Model</title>
  <script>
    function showLoading() {
      // Hide the form and display the loading message.
      document.getElementById("formDiv").style.display = "none";
      document.getElementById("loadingDiv").style.display = "block";
    }
  </script>
</head>
<body>
  <h1>Train Model</h1>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul style="color: green;">
      {% for message in messages %}
        <li>{{ message }}</li>
      {% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
  <div id="formDiv">
    <form method="post" enctype="multipart/form-data" onsubmit="showLoading();">
      <label for="cloth_name">Cloth Name/Label:</label>
      <input type="text" name="cloth_name" id="cloth_name" required><br><br>
      <label for="train_images">Select training images (you can select multiple):</label>
      <input type="file" name="train_images" id="train_images" multiple accept="image/*"><br><br>
      <input type="submit" value="Upload and Train">
    </form>
  </div>
  <div id="loadingDiv" style="display: none;">
    <p><strong>Training images, please wait...</strong></p>
  </div>
  <br>
  <a href="/">Back to Home</a>
</body>
</html>
    ''')

# -------- Prediction Route --------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Prediction page:
      - Users upload a test image.
      - The image's embedding is computed and compared to the stored training embeddings using FAISS.
      - The cloth name corresponding to the closest match is returned.
    """
    result = None
    if request.method == 'POST':
        file = request.files.get("test_image")
        if not file or file.filename == "":
            flash("Please upload an image for prediction!")
            return redirect(url_for('predict'))
        try:
            image_bytes = file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            user_embedding = compute_embedding(pil_image)
            if index.ntotal == 0:
                flash("No training images available. Please train the model first!")
                return redirect(url_for('predict'))
            distances, indices = index.search(user_embedding, 1)
            match_index = indices.flatten()[0]
            result = product_names[match_index]
        except Exception as e:
            flash(f"Error processing image: {e}")
            return redirect(url_for('predict'))
    
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
  <title>Predict Clothing</title>
</head>
<body>
  <h1>Predict Clothing</h1>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul style="color: red;">
      {% for message in messages %}
        <li>{{ message }}</li>
      {% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
  <form method="post" enctype="multipart/form-data">
    <label for="test_image">Select an image for prediction:</label>
    <input type="file" name="test_image" id="test_image" accept="image/*" required><br><br>
    <input type="submit" value="Predict">
  </form>
  {% if result %}
    <h2>Predicted Cloth Name: {{ result }}</h2>
  {% endif %}
  <br>
  <a href="/">Back to Home</a>
</body>
</html>
    ''', result=result)

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    # Attempt to load previously saved training data
    load_training_data()
    # Disable the reloader so that global variables persist
    app.run(debug=True, use_reloader=False)
