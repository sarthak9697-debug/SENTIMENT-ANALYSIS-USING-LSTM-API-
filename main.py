import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pickle
import re
import string
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = Path(__file__).with_name("model.pkl")

app = FastAPI(
    title="💬 LSTM Sentiment Analysis API",
    version="1.0.0",
    description="API for the LSTM sentiment-analysis assignment. Predicts Negative, Neutral, or Positive sentiment from text."
)

SAMPLE_TEXTS = {
    1: "Donald Trump performed really well in the debate",
    2: "I really liked the candidate's answer tonight",
    3: "That was a terrible answer and very disappointing",
    4: "The candidate performed badly and avoided the question",
    5: "The debate started at nine tonight",
    6: "The candidates discussed several policy topics",
}

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Tweet or short sentence to classify.",
        examples=["The candidate gave a strong answer tonight."]
    )

class SentimentResponse(BaseModel):
    predicted_sentiment: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str

def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run the notebook deployment cell first.")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = model_from_json(bundle["model_json"])
    model.set_weights(bundle["model_weights"])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, bundle

try:
    model, bundle = load_bundle()
except Exception as exc:
    model = None
    bundle = None
    LOAD_ERROR = str(exc)

def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return " ".join(w for w in text.split() if w not in stop_words)

def prepare_text(text):
    cleaned = clean_text(text, set(bundle["stop_words"]))
    word_index = bundle["tokenizer"]["word_index"]
    oov_token = bundle["tokenizer"].get("oov_token")
    oov_index = word_index.get(oov_token, 0) if oov_token else 0
    sequence = [word_index.get(w, oov_index) for w in cleaned.split()]
    return pad_sequences([sequence], maxlen=int(bundle["max_length_sequence"])), cleaned

@app.get("/")
def root():
    if model is None:
        return {"status": "error", "message": LOAD_ERROR}
    return {
        "status": "online",
        "message": "LSTM Sentiment Analysis API is running.",
        "documentation_url": "/docs",
        "classes": bundle["label_classes"],
    }

@app.get("/samples")
def samples():
    return {"samples": SAMPLE_TEXTS}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=LOAD_ERROR)
    try:
        x, cleaned = prepare_text(request.text)
        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = bundle["label_classes"][idx]
        return {
            "predicted_sentiment": label,
            "confidence": float(probs[idx]),
            "probabilities": {
                cls: float(prob)
                for cls, prob in zip(bundle["label_classes"], probs)
            },
            "cleaned_text": cleaned,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}")
