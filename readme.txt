1. Login: euphoriainfotech.apps@gmail.com

2. Goto Google COLAB.
https://colab.research.google.com/drive/1TdJhI3D4nefQsshnaGcj95F-zDXXlPdd?usp=sharing
https://colab.research.google.com/drive/1TdJhI3D4nefQsshnaGcj95F-zDXXlPdd#scrollTo=a9ifoqP5tp9d

3. Upload images to "known_faces" Folder.

4. Generate "embeddings.pkl" using 1 to 6 Commands.

5. Copy "embeddings.pkl" File and Paste it to Github Repo:
https://github.com/EuphoriaInfotech/AIFaceMatchingTesting

6. Now open render.com
https://dashboard.render.com/web/srv-d4oufdkhg0os73cfitdg/deploys/dep-d4ouj12dbo4c73ftebq0

7. Set Environment_Settings:
https://dashboard.render.com/web/srv-d4oufdkhg0os73cfitdg/env
PYTHON_VERSION  :   3.10.13

Deploy > Deploy with last commit


GOOGLE COLAB CODE(For Generating embeddings.pkl File):-
--------------------------------------------------------
from google.colab import files
uploaded = files.upload()
********************************************************
import os, shutil
for file in uploaded:
    shutil.move(file, f"known_faces/{file}")
********************************************************
!pip install fastapi uvicorn opencv-python mediapipe deepface pyngrok
********************************************************
create_embeddings.py
------------------------------------
from deepface import DeepFace
import os
import pickle

KNOWN_DIR = "known_faces"
EMBED_FILE = "embeddings.pkl"

db = {}

if not os.path.exists(KNOWN_DIR):
    raise RuntimeError("❌ known_faces folder not found")

for file in os.listdir(KNOWN_DIR):
    path = os.path.join(KNOWN_DIR, file)

    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    name = os.path.splitext(file)[0]

    print(f"Processing: {name}")

    try:
        embedding = DeepFace.represent(
            img_path=path,
            model_name="Facenet512",
            enforce_detection=False
        )[0]["embedding"]

        db[name] = embedding

    except Exception as e:
        print(f"❌ Failed for {name}: {e}")

# ✅ SAVE FILE (THIS WAS MISSING IN YOUR CODE)
with open(EMBED_FILE, "wb") as f:
    pickle.dump(db, f)

print("✅ embeddings.pkl created successfully!")

********************************************************
!python create_embeddings.py
********************************************************
import pickle

with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

print(db.keys())
********************************************************
from google.colab import files
files.download("embeddings.pkl")


