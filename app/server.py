from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io, json, os

app = FastAPI(title="Cat vs Dog Classifier")

MODEL_PATH = "models/best.keras"
LABEL_MAP_PATH = "models/label_map.json"

# Load artifacts once
model = load_model(MODEL_PATH)
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
class_names = {v: k for k, v in label_map.items()}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)

    preds = model.predict(x)
    class_id = int(np.argmax(preds))
    prob = float(np.max(preds))
    return JSONResponse({"label": class_names[class_id], "probability": prob})
