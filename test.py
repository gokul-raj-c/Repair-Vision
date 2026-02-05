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
CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "car_check_model")
car_model = tf.saved_model.load(CAR_MODEL_PATH)
infer_car = car_model.signatures["serving_default"]

# Damage Detection
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_check_model")
damage_model = tf.saved_model.load(DAMAGE_MODEL_PATH)
infer_damage = damage_model.signatures["serving_default"]

# YOLO Damage Parts Model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
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
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])

            detected_parts.append({
                "part": yolo_model.names[cls_id],
                "confidence": int(conf * 100)
            })

            # Draw ONLY bounding box (no text)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)



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







"-----------------------------------------------------------------------------------------------------------------------"



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
CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "car_check_model")
car_model = tf.saved_model.load(CAR_MODEL_PATH)
infer_car = car_model.signatures["serving_default"]

# Damage Yes/No
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_check_model")
damage_model = tf.saved_model.load(DAMAGE_MODEL_PATH)
infer_damage = damage_model.signatures["serving_default"]

# YOLO Damage Parts Model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
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

# ---------------- SEVERITY COLOR ---------------- #

def get_severity_color(conf):
    if conf >= 0.75:
        return (0, 0, 255)      # 🔴 High
    elif conf >= 0.50:
        return (0, 255, 255)    # 🟡 Medium
    else:
        return (0, 255, 0)      # 🟢 Low

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

    if (1 - car_pred) < 0.5:
        return jsonify({"status": "no_car"})

    # ---------- DAMAGE DETECTION ---------- #
    damage_img = preprocess_damage_image(input_path)
    damage_pred = list(
        infer_damage(keras_tensor=tf.constant(damage_img, dtype=tf.float32)).values()
    )[0].numpy()[0][0]

    if (1 - damage_pred) < 0.5:
        return jsonify({"status": "no_damage"})

    # ---------- YOLO DAMAGE PART DETECTION ---------- #

    results = yolo_model.predict(
        source=input_path,
        conf=0.45,
        iou=0.6,       # better suppression
        device="cpu",
        save=False,
        verbose=False
    )

    img = cv2.imread(input_path)
    best_boxes = {}

    # 🔹 Step 1: Keep best box per class
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            cls_id = int(box.cls[0])

            if cls_id not in best_boxes or conf > best_boxes[cls_id]["conf"]:
                best_boxes[cls_id] = {
                    "box": box,
                    "conf": conf
                }

    if not best_boxes:
        return jsonify({"status": "damage_but_part_not_detected"})

    detected_parts = []

    # 🔹 Step 2: Draw boxes + collect parts
    for cls_id, data in best_boxes.items():
        box = data["box"]
        conf = data["conf"]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detected_parts.append({
            "part": yolo_model.names[cls_id],
            "confidence": int(conf * 100)
        })

        color = get_severity_color(conf)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    cv2.imwrite(output_path, img)

    session["output_image"] = f"outputs/{filename}"
    session["detected_parts"] = detected_parts

    return jsonify({"status": "damage_detected"})

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)



"-----------------------------------------------------------------------------------------------------------------------"



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
CAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "car_check_model")
car_model = tf.saved_model.load(CAR_MODEL_PATH)
infer_car = car_model.signatures["serving_default"]

# Damage Yes/No
DAMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_check_model")
damage_model = tf.saved_model.load(DAMAGE_MODEL_PATH)
infer_damage = damage_model.signatures["serving_default"]

# YOLO Damage Parts Model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_best.pt")
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

# ---------------- SEVERITY COLOR ---------------- #

def get_severity_color(conf):
    if conf >= 0.75:
        return (0, 0, 255)      # 🔴 High
    elif conf >= 0.50:
        return (0, 255, 255)    # 🟡 Medium
    else:
        return (0, 255, 0)      # 🟢 Low

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

    if (1 - car_pred) < 0.5:
        return jsonify({"status": "no_car"})

    # ---------- DAMAGE DETECTION ---------- #
    damage_img = preprocess_damage_image(input_path)
    damage_pred = list(
        infer_damage(keras_tensor=tf.constant(damage_img, dtype=tf.float32)).values()
    )[0].numpy()[0][0]

    if (1 - damage_pred) < 0.5:
        return jsonify({"status": "no_damage"})

    # ---------- YOLO DAMAGE PART DETECTION ---------- #

    results = yolo_model.predict(
        source=input_path,
        conf=0.6,
        iou=0.6,
        max_det=20,
        device="cpu",
        save=False,
        verbose=False
    )

    img = cv2.imread(input_path)
    best_boxes = {}

    # 🔹 Step 1: Keep BEST box per class
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.6:
                continue

            cls_id = int(box.cls[0])

            if cls_id not in best_boxes or conf > best_boxes[cls_id]["conf"]:
                best_boxes[cls_id] = {
                    "box": box,
                    "conf": conf
                }

    if not best_boxes:
        return jsonify({"status": "damage_but_part_not_detected"})

    # 🔹 Step 2: Sort by confidence & keep Top-K
    MAX_PARTS = 3
    sorted_boxes = sorted(
        best_boxes.items(),
        key=lambda x: x[1]["conf"],
        reverse=True
    )[:MAX_PARTS]

    detected_parts = []

    # 🔹 Step 3: Draw ONLY Top-K boxes
    for cls_id, data in sorted_boxes:
        box = data["box"]
        conf = data["conf"]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detected_parts.append({
            "part": yolo_model.names[cls_id],
            "confidence": int(conf * 100)
        })

        color = get_severity_color(conf)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

    cv2.imwrite(output_path, img)

    session["output_image"] = f"outputs/{filename}"
    session["detected_parts"] = detected_parts

    return jsonify({"status": "damage_detected"})

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    app.run(debug=True)




"-----------------------------------------------------------------------------------------------------------------------"



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
app.secret_key = "repairvision_secret"

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

# YOLO models
yolo_parts  = YOLO(os.path.join(BASE_DIR, "models/parts_best.pt"))   # Scratch/Dent
yolo_damage = YOLO(os.path.join(BASE_DIR, "models/damage_best.pt"))  # Bumper/Hood

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

    # ---------- YOLO PARTS (DAMAGE TYPE) ---------- #
    img = cv2.imread(input_path)
    H, W = img.shape[:2]

    damage_results = yolo_parts.predict(
        source=input_path,
        conf=0.5,
        iou=0.6,
        device="cpu",
        verbose=False
    )

    detections = []

    for r in damage_results:
        for box in r.boxes:
            d_conf = float(box.conf[0])
            if d_conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            damage_cls = int(box.cls[0])
            damage_name = yolo_parts.names[damage_cls]

            # ---------- CROP DAMAGE REGION ---------- #
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ---------- YOLO DAMAGE (CAR PART) ---------- #
            part_result = yolo_damage.predict(
                source=crop,
                conf=0.4,
                iou=0.6,
                device="cpu",
                verbose=False
            )

            part_name = "Unknown"
            part_conf = 0

            if len(part_result[0].boxes) > 0:
                pbox = part_result[0].boxes[0]
                part_cls = int(pbox.cls[0])
                part_conf = int(float(pbox.conf[0]) * 100)
                part_name = yolo_damage.names[part_cls]

            # ---------- DRAW BOX ---------- #
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            detections.append({
                "damage": damage_name,
                "damage_conf": int(d_conf * 100),
                "part": part_name,
                "part_conf": part_conf
            })

    if not detections:
        return jsonify({"status": "damage_but_part_not_detected"})

    cv2.imwrite(output_path, img)

    session["output_image"] = f"outputs/{filename}"
    session["detected_parts"] = detections

    return jsonify({"status": "damage_detected"})

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)



