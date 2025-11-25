from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from detector import get_pose_landmarks, get_face_landmarks
from orientation import combine_orientation

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", result=None, preview=None)

@app.route("/predict-web", methods=["POST"])
def predict_web():

    file = request.files.get("image", None)
    if file is None or file.filename == "":
        return render_template("index.html", result="No image uploaded", preview=None)

    # Read uploaded image bytes directly (NO SAVING)
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return render_template("index.html", result="Invalid image", preview=None)

    h, w = img.shape[:2]

    pose_landmarks = get_pose_landmarks(img)

    face_output = get_face_landmarks(img)
    face_landmarks = None
    face_count = 0
    if face_output:
        face_landmarks, face_count = face_output

    final = combine_orientation(pose_landmarks, face_landmarks, face_count, w)

    # Convert uploaded image to Base64 to show preview on SAME PAGE
    import base64
    preview_img = base64.b64encode(img_bytes).decode("utf-8")
    preview_src = f"data:image/jpeg;base64,{preview_img}"

    return render_template("index.html", result=final, preview=preview_src)


@app.route("/predict", methods=["POST"])
def predict_api():

    file = request.files.get("image", None)
    if file is None:
        return jsonify({"error": "No image provided"})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    h, w = img.shape[:2]

    pose_landmarks = get_pose_landmarks(img)

    face_output = get_face_landmarks(img)
    face_landmarks = None
    face_count = 0
    if face_output:
        face_landmarks, face_count = face_output

    final = combine_orientation(pose_landmarks, face_landmarks, face_count, w)

    return jsonify({"orientation": final})


if __name__ == "__main__":
    app.run(debug=True)
