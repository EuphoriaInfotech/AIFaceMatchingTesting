from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np, pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine

app = FastAPI()

with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

@app.post("/match-group/")
async def match_group(image: UploadFile = File(...)):
    img_bytes = await image.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = DeepFace.extract_faces(img, enforce_detection=False)

    results = []
    for face in faces:
        emb = DeepFace.represent(face["face"], model_name="Facenet", enforce_detection=False)[0]["embedding"]

        name = "Unknown"
        min_dist = 10

        for person, db_emb in db.items():
            dist = cosine(db_emb, emb)
            if dist < min_dist and dist < 0.4:
                min_dist = dist
                name = person

        results.append(name)

    return {
        "total_faces": len(results),
        "matched_names": results
    }
