from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit uploads to 50MB

# Load model globally at startup
MODEL_PATH = 'model1.h5'
model = load_model(MODEL_PATH)

# Setup upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Reviews file
REVIEWS_FILE = 'reviews.json'


def load_reviews():
    try:
        with open(REVIEWS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_review(review):
    reviews = load_reviews()
    reviews.append(review)
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = "Prediction: Not available"
    uploaded_image = None
    reviews = load_reviews()

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)
            uploaded_image = f"/static/uploads/{uploaded_file.filename}"

            # Preprocess the image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict
            pred = model.predict(img_array)
            predicted_class = "Real" if pred[0][0] > 0.35 else "Fake"
            prediction = f"Prediction: ✅ {predicted_class} Logo" if predicted_class == "Real" else f"Prediction: ❌ Possibly Fake Logo"

    return render_template("index.html", prediction=prediction, uploaded_image=uploaded_image, reviews=reviews)


@app.route("/submit_review", methods=["POST"])
def submit_review():
    review_text = request.form["review"]
    stars = request.form.get("stars", "5")
    save_review({"text": review_text, "stars": stars})
    return index()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True)




         
