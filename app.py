# ---------------- IMPORTS ---------------- #
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# ---------------- APP INIT ---------------- #
app = Flask(__name__)
app.secret_key = "damage_detection_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- MODEL PATHS ---------------- #

# Car vs No-Car
CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "car_detect_model")
car_model = tf.saved_model.load(CAR_MODEL_PATH)
infer_car = car_model.signatures["serving_default"]

# Damage Detection
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_detect_model")
damage_model = tf.saved_model.load(DAMAGE_MODEL_PATH)
infer_damage = damage_model.signatures["serving_default"]

# YOLO Damage Parts Model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_identification_model.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

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
        "res.html",
        original_image=session.get("original_image"),
        output_image=session.get("output_image"),
        detected_parts=session.get("detected_parts")
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
    img = cv2.resize(img, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- MAIN PREDICTION ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"status": "error"})

    file = request.files["file"]
    filename = secure_filename(file.filename)

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    file.save(input_path)

    # ---------- CAR DETECTION ---------- #
    car_img = preprocess_car_image(input_path)
    car_pred = list(
        infer_car(keras_tensor=tf.constant(car_img, dtype=tf.float32)).values()
    )[0].numpy()[0][0]

    car_confidence = 1 - car_pred
    print("car confidence:",car_confidence)

    if car_confidence < 0.5:
        return jsonify({"status": "no_car"})

    # ---------- DAMAGE DETECTION ---------- #
    damage_img = preprocess_damage_image(input_path)
    damage_pred = list(
        infer_damage(keras_tensor=tf.constant(damage_img, dtype=tf.float32)).values()
    )[0].numpy()[0][0]

    damage_confidence = 1 - damage_pred
    print("damage confidence:",damage_confidence)

    if damage_confidence < 0.5:
        return jsonify({"status": "no_damage"})

    # ---------- YOLO DAMAGE PART DETECTION ---------- #

    #device = "cuda" if torch.cuda.is_available() else "cpu"


    results = yolo_model.predict(
        source=input_path,
        conf=0.45,
        iou=0.4,
        device="cpu",
        save=False,
        verbose=False
    )
    img = cv2.imread(input_path)
    detected_parts = []


    detected = False

    for r in results:
        for box in r.boxes:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            detected_parts.append({
                "part": yolo_model.names[cls_id],
                "confidence": int(conf * 100)
            })

            label = f"{yolo_model.names[cls_id]} {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text
            label_text = f"{yolo_model.names[cls_id].capitalize()} ({int(conf*100)}%)"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Ensure label stays inside image
            y_label = max(y1, text_height + 10)

            # Draw filled rectangle behind text
            cv2.rectangle(
                img,
                (x1, y_label - text_height - 10),
                (x1 + text_width + 6, y_label),
                (0, 255, 0),
                -1
            )

            # Draw label text (black for contrast)
            cv2.putText(
                img,
                label_text,
                (x1 + 3, y_label - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )


    if not detected:
        return jsonify({"status": "damage_but_part_not_detected"})

    cv2.imwrite(output_path, img)

    session["original_image"] = f"uploads/{filename}"
    session["output_image"] = f"outputs/{filename}"
    session["status"] = "Damage detected with affected parts highlighted"
    session["detected_parts"] = detected_parts


    return jsonify({"status": "damage_detected"})

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    app.run(debug=True)
