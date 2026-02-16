# ---------------- IMPORTS ---------------- #
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from flask import send_file
from flask_mail import Mail, Message
import random

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------------- APP INIT ---------------- #
app = Flask(__name__)
app.secret_key = "repairvision_secret"


# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "teamgarage4web@gmail.com"    
app.config['MAIL_PASSWORD'] = "tptnqdqdzmojldoe"        
app.config['MAIL_DEFAULT_SENDER'] = "your_email@gmail.com"

mail = Mail(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static/outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- MODELS ---------------- #

# Car vs No-Car
car_model = tf.saved_model.load(os.path.join(BASE_DIR, "models/car_check_model"))
infer_car = car_model.signatures["serving_default"]

# Damage Yes/No
damage_model = tf.saved_model.load(os.path.join(BASE_DIR, "models/damage_check_model"))
infer_damage = damage_model.signatures["serving_default"]

# YOLO MODELS
yolo_damage = YOLO(os.path.join(BASE_DIR, "models/damage_best.pt"))  # body part detector
yolo_parts  = YOLO(os.path.join(BASE_DIR, "models/parts_best.pt"))   # scratch/dent detector

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
        output_image=session.get("output_image"),
        detected_parts=session.get("detected_parts")
    )

# ---------------- PREPROCESS ---------------- #

def preprocess_tf(img_path):
    img = cv2.imread(img_path)
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

    input_path  = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    file.save(input_path)

    # ---------- CAR CHECK ---------- #
    car_pred = list(
        infer_car(keras_tensor=tf.constant(preprocess_tf(input_path))).values()
    )[0].numpy()[0][0]

    if (1 - car_pred) < 0.5:
        return jsonify({"status": "no_car"})

    # ---------- DAMAGE CHECK ---------- #
    damage_pred = list(
        infer_damage(keras_tensor=tf.constant(preprocess_tf(input_path))).values()
    )[0].numpy()[0][0]

    if (1 - damage_pred) < 0.5:
        return jsonify({"status": "no_damage"})

    # ---------- STEP 1: DETECT CAR PART (BUMPER/HOOD etc) ---------- #
    img = cv2.imread(input_path)

    part_results = yolo_damage.predict(
        source=input_path,
        conf=0.6,
        iou=0.6,
        device="cpu",
        verbose=False
    )

    detections = []

    # ---------- LOOP EACH CAR PART ---------- #
    for r in part_results:
        for box in r.boxes:

            part_conf = float(box.conf[0])
            if part_conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            part_cls = int(box.cls[0])
            part_name = yolo_damage.names[part_cls]

            # ---------- CROP CAR PART REGION ---------- #
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ---------- STEP 2: DETECT DAMAGE TYPE INSIDE PART ---------- #
            damage_results = yolo_parts.predict(
                source=crop,
                conf=0.45,
                iou=0.5,
                device="cpu",
                verbose=False
            )

            # ❗ if NO damage detected → skip this part
            if len(damage_results[0].boxes) == 0:
                continue

            # take BEST damage box
            best_damage = max(
                damage_results[0].boxes,
                key=lambda b: float(b.conf[0])
            )

            damage_conf = float(best_damage.conf[0])
            if damage_conf < 0.45:
                continue

            damage_cls = int(best_damage.cls[0])
            damage_name = yolo_parts.names[damage_cls]

            # ---------- DRAW BOX ON ORIGINAL IMAGE ---------- #
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)

            detections.append({
                "part": part_name,
                "part_conf": int(part_conf * 100),
                "damage": damage_name,
                "damage_conf": int(damage_conf *100)
            })

    if not detections:
        return jsonify({"status": "damage_but_part_not_detected"})

    cv2.imwrite(output_path, img)

    session["output_image"] = f"outputs/{filename}"
    session["detected_parts"] = detections

    return jsonify({"status": "damage_detected"})


# ---------------- DOWNLOAD PDF ---------------- #

@app.route("/download_pdf")
def download_pdf():

    output_image = session.get("output_image")
    detected_parts = session.get("detected_parts")

    if not output_image:
        return "No result to generate PDF"

    pdf_path = os.path.join(OUTPUT_FOLDER, "damage_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # ---------- TITLE ----------
    c.setFont("Helvetica-Bold", 20)
    c.drawString(180, height - 50, "Repair Vision Damage Report")

    # ---------- DATE ----------
    c.setFont("Helvetica", 11)
    date_str = datetime.now().strftime("%d %B %Y, %I:%M %p")
    c.drawString(50, height - 80, f"Generated on: {date_str}")

    # ---------- IMAGE ----------
    img_path = os.path.join("static", output_image).replace("\\", "/")
    img_path = os.path.join(BASE_DIR, img_path)

    c.drawImage(img_path, 50, height - 420, width=300, height=300)

    # ---------- DAMAGE DETAILS ----------
    y = height - 450
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Detected Damages:")
    y -= 20

    c.setFont("Helvetica", 12)

    if detected_parts:
        for d in detected_parts:
            text = f"{d['part']} ({d['part_conf']}%)  →  {d['damage']} ({d['damage_conf']}%)"
            c.drawString(60, y, text)
            y -= 18
            if y < 80:
                c.showPage()
                y = height - 50
    else:
        c.drawString(60, y, "No damage detected")

    # ---------- FOOTER ----------
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(180, 30, "Generated by Repair Vision AI System")

    c.save()

    return send_file(pdf_path, as_attachment=True)


# ---------------- SEND EMAIL ---------------- #

@app.route("/send_report", methods=["POST"])
def send_report():

    try:
        email = request.form.get("email")
        if not email:
            return jsonify({"status": "no_email"})

        output_image = session.get("output_image")
        detected_parts = session.get("detected_parts")

        if not output_image:
            return jsonify({"status": "no_data"})

        # ---------- CREATE PDF AUTOMATICALLY ----------

        pdf_path = os.path.join(OUTPUT_FOLDER, "damage_report.pdf")

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(180, height - 50, "Repair Vision Damage Report")

        # Date
        c.setFont("Helvetica", 11)
        date_str = datetime.now().strftime("%d %B %Y, %I:%M %p")
        c.drawString(50, height - 80, f"Generated on: {date_str}")

        # Image
        img_path = os.path.join(BASE_DIR, "static", output_image)
        c.drawImage(img_path, 50, height - 420, width=300, height=300)

        # Damage list
        y = height - 450
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Detected Damages:")
        y -= 25

        c.setFont("Helvetica", 12)

        if detected_parts:
            for d in detected_parts:
                text = f"{d['part']} ({d['part_conf']}%) → {d['damage']} ({d['damage_conf']}%)"
                c.drawString(60, y, text)
                y -= 18
                if y < 60:
                    c.showPage()
                    y = height - 50
        else:
            c.drawString(60, y, "No damage detected")

        c.save()

        # ---------- SEND MAIL ----------
        msg = Message(
            subject="🚗 Repair Vision Damage Report",
            recipients=[email],
            body="Your AI car damage report is attached."
        )

        with open(pdf_path, "rb") as f:
            msg.attach("Damage_Report.pdf", "application/pdf", f.read())

        mail.send(msg)

        return jsonify({"status": "sent"})

    except Exception as e:
        print("MAIL ERROR:", e)
        return jsonify({"status": "error"})



# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)
