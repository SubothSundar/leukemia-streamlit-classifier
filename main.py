import io
import os
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf


MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.txt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))


def load_labels(path: str) -> List[str]:
	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f:
			return [line.strip() for line in f if line.strip()]
	# Default fallback
	return ["L1", "L2", "L3"]


def load_model(path: str):
	if not os.path.exists(path):
		return None
	return tf.keras.models.load_model(path)


def preprocess_image_bytes(file_bytes: bytes, target_size: int = 224) -> np.ndarray:
	image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
	image = image.resize((target_size, target_size))
	array = np.asarray(image).astype("float32") / 255.0
	array = np.expand_dims(array, axis=0)
	return array


app = FastAPI(title="Leukemia Classification API", version="1.0")


@app.on_event("startup")
def _startup_load():
	global model, class_labels
	class_labels = load_labels(LABELS_PATH)
	model = load_model(MODEL_PATH)

# Serve React UI under /app
app.mount("/app", StaticFiles(directory="frontend", html=True), name="react_app")


@app.get("/", response_class=HTMLResponse)
def index_page():
	return RedirectResponse(url="/app", status_code=307)


@app.get("/healthz")
def health():
	status = {
		"model_loaded": bool(model is not None),
		"labels": class_labels,
		"img_size": IMG_SIZE,
	}
	return JSONResponse(status)


@app.get("/favicon.ico")
def favicon():
	# Avoid 404 in logs; browsers will accept empty favicon
	return Response(status_code=204, media_type="image/x-icon")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	if model is None:
		raise HTTPException(status_code=503, detail="Model not loaded. Upload model.h5 and labels.txt.")

	contents = await file.read()
	try:
		input_tensor = preprocess_image_bytes(contents, target_size=IMG_SIZE)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

	probs = model.predict(input_tensor)[0].tolist()
	idx = int(np.argmax(probs))
	pred_label = class_labels[idx] if idx < len(class_labels) else str(idx)

	return {"prediction": pred_label, "probabilities": dict(zip(class_labels, probs))}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), reload=False)


