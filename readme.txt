1. Login: euphoriainfotech.apps@gmail.com

2. Goto Google COLAB.
https://colab.research.google.com/drive/1TdJhI3D4nefQsshnaGcj95F-zDXXlPdd#scrollTo=a9ifoqP5tp9d

3. Upload images to "known_faces" Folder.

4. Rum "Generate embeddings.pkl File" Command.

5. Copy "embeddings.pkl" File and Paste it to Github Repo:
https://github.com/EuphoriaInfotech/AIFaceMatchingTesting

6. Now open render.com
https://dashboard.render.com/web/srv-d4oufdkhg0os73cfitdg/deploys/dep-d4ouj12dbo4c73ftebq0

Deploy > Deploy with last commit


GOOGLE COLAB CODE(For Generating embeddings.pkl File):-
--------------------------------------------------------
from google.colab import files
uploaded = files.upload()


import os, shutil
for file in uploaded:
    shutil.move(file, f"known_faces/{file}")


!pip install fastapi uvicorn opencv-python mediapipe deepface pyngrok


from deepface import DeepFace
import os
import pickle

KNOWN_DIR = "known_faces"
db = {}

for file in os.listdir(KNOWN_DIR):
    path = f"{KNOWN_DIR}/{file}"
    name = os.path.splitext(file)[0]
    embedding = DeepFace.represent(path, model_name="Facenet")[0]["embedding"]
    db[name] = embedding

with open("embeddings.pkl", "wb") as f:
    pickle.dump(db, f)

print("✅ Face embeddings saved")





