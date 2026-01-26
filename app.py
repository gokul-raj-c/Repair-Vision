# ---------------- IMPORTS ---------------- #
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

# ---------------- APP INIT ---------------- #

app = Flask(__name__)
app.secret_key = "damage_detection_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- MODEL PATHS ---------------- #

# Car vs No-Car (SavedModel)
CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "car_detect_model")
car_model = tf.saved_model.load(CAR_MODEL_PATH)
infer_car = car_model.signatures["serving_default"]

# Damage Detection (SavedModel)
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_detect_model")
damage_model = tf.saved_model.load(DAMAGE_MODEL_PATH)
infer_damage = damage_model.signatures["serving_default"]

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/result")
def result():
    return render_template(
        "result.html",
        original_image=session.get("original_image"),
        output_image=session.get("output_image")
    )

# ---------------- IMAGE PREPROCESSING ---------------- #

def preprocess_car_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_damage_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))   # ✅ MUST MATCH TRAINING
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)



# ---------------- MAIN PREDICTION ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"status": "error"})

    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ---------- CAR DETECTION ---------- #
    car_img = preprocess_car_image(filepath)
    car_outputs = infer_car(
        keras_tensor=tf.constant(car_img, dtype=tf.float32)
    )
    car_pred = list(car_outputs.values())[0].numpy()[0][0]

    car_confidence = 1 - car_pred

    if car_confidence < 0.4:
        return jsonify({"status": "no_car"})

    # ---------- DAMAGE DETECTION ---------- #
    damage_img = preprocess_damage_image(filepath)
    damage_outputs = infer_damage(
        keras_tensor=tf.constant(damage_img, dtype=tf.float32)
    )
    damage_pred = list(damage_outputs.values())[0].numpy()[0][0]

    damage_confidence = 1 - damage_pred
    print("Damage confidence:", damage_confidence)

    if damage_confidence < 0.4:
        return jsonify({"status": "no_damage"})

    return jsonify({"status": "damage_detected"})

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
