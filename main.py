from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine
import uuid
import os

app = FastAPI()

# ✅ Load embeddings safely
if not os.path.exists("embeddings.pkl"):
    raise RuntimeError("❌ embeddings.pkl not found in project root")

with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# ✅ Health check for Render
@app.get("/")
def root():
    return {"status": "Face Matching API is running ✅"}

# ✅ Group face matching API
@app.post("/match-group/")
async def match_group(image: UploadFile = File(...)):

    # Read image
    img_bytes = await image.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # ✅ Detect faces using RetinaFace (NO dlib)
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend="retinaface",
        enforce_detection=False
    )

    results = []

    for face in faces:
        face_img = face["face"]

        # ✅ Generate embedding
        rep = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet512",
