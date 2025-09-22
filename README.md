# Leukemia Image Classifier (Streamlit)

Simple CNN for 3-class leukemia image classification (L1, L2, L3). Train with TensorFlow, serve via Streamlit.

## Local run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app.
3. Repo: YOUR_USER/leukemia-streamlit-classifier, Branch: main
4. Main file: `streamlit_app.py`
5. (Optional) Environment variables:
   - `MODEL_PATH=model2.keras`
   - `LABELS_PATH=labels.txt`
   - `IMG_SIZE=224`
6. Deploy and test.

## Train a model
```powershell
python train.py  # uses MyData/L1,L2,L3; writes model2.keras and labels.txt
```

## Layout
- `train.py` – training pipeline
- `streamlit_app.py` – Streamlit UI
- `main.py` – FastAPI API (optional)
- `requirements.txt` – dependencies
- `model2.keras`, `labels.txt` – trained model + labels

Notes: Keep `MyData/` out of git (see `.gitignore`).
