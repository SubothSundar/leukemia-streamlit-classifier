import io
import os
from typing import List

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = os.environ.get("MODEL_PATH", "model2.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.txt")
IMG_SIZE = int(os.environ.get("IMG_SIZE", "224"))
MODEL_URL = os.environ.get("MODEL_URL")


def _download_model_if_needed(url: str | None, target_path: str) -> None:
	if not url:
		return
	if os.path.exists(target_path):
		return
	try:
		with st.status("Downloading model...", expanded=False):
			if "drive.google.com" in url:
				# Use gdown for Google Drive (handles confirmation tokens)
				try:
					import gdown  # lazy import to avoid hard dependency when MODEL_URL is empty
				except Exception as e:
					st.error("gdown is required for Google Drive downloads. Install it or provide a direct URL.")
					raise
				gdown.download(url, target_path, quiet=False, fuzzy=True)
			else:
				with requests.get(url, stream=True, timeout=60) as r:
					r.raise_for_status()
					with open(target_path, "wb") as f:
						for chunk in r.iter_content(chunk_size=1024 * 1024):
							if chunk:
								f.write(chunk)
	except Exception as e:
		st.error(f"Failed to download model: {e}")


@st.cache_resource(show_spinner=False)
def _resolve_model_path(path: str) -> str | None:
	candidates = [path, "model2.keras"]
	for p in candidates:
		if p and os.path.exists(p):
			return p
	return None


@st.cache_resource(show_spinner=False)
def load_model(path: str):
	if MODEL_URL:
		_download_model_if_needed(MODEL_URL, path)
	resolved = _resolve_model_path(path)
	if not resolved:
		return None
	return tf.keras.models.load_model(resolved)


@st.cache_resource(show_spinner=False)
def load_labels(path: str) -> List[str]:
	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f:
			return [line.strip() for line in f if line.strip()]
	return ["L1", "L2", "L3"]


def preprocess_image(image: Image.Image, target_size: int) -> np.ndarray:
	image = image.convert("RGB").resize((target_size, target_size))
	array = np.asarray(image).astype("float32") / 255.0
	array = np.expand_dims(array, axis=0)
	return array


def main():
	st.set_page_config(page_title="Leukemia Classifier", page_icon="🧬", layout="centered")
	st.title("AI-Powered Leukemia Image Classification")
	st.caption("Classes: L1, L2, L3")

	model = load_model(MODEL_PATH)
	labels = load_labels(LABELS_PATH)

	with st.sidebar:
		st.subheader("Settings")
		resolved = _resolve_model_path(MODEL_PATH)
		st.write(f"Model: {'Loaded' if model is not None else 'Not found'}")
		st.write(f"MODEL_PATH: {MODEL_PATH}")
		st.write(f"Resolved: {resolved or 'N/A'}")
		st.write(f"Image size: {IMG_SIZE}×{IMG_SIZE}")
		st.write("Labels: " + ", ".join(labels))

	uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"]) 
	if uploaded is not None:
		image = Image.open(uploaded)
		st.image(image, caption="Uploaded image", use_column_width=True)

		if model is None:
			st.error("Model not loaded. Ensure model2.keras is present or MODEL_URL is set.")
			return

		if st.button("Predict"):
			with st.spinner("Running inference..."):
				inputs = preprocess_image(image, IMG_SIZE)
				probs = model.predict(inputs)[0]
				idx = int(np.argmax(probs))
				pred_label = labels[idx] if idx < len(labels) else str(idx)

				st.success(f"Prediction: {pred_label}")
				st.write("Probabilities:")
				for label, p in sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True):
					st.progress(min(max(p, 0.0), 1.0))
					st.write(f"{label}: {p*100:.1f}%")


if __name__ == "__main__":
	main()


