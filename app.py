"""
Student Performance Predictor – Flask REST API
Author: Hemant Sahu
Run: python src/app.py
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="../templates")

# ── Load model artifacts ──────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE, "models/best_model.pkl"),   "rb") as f: model        = pickle.load(f)
with open(os.path.join(BASE, "models/scaler.pkl"),       "rb") as f: scaler       = pickle.load(f)
with open(os.path.join(BASE, "models/feature_cols.pkl"), "rb") as f: feature_cols = pickle.load(f)

# ── Encoding maps (must match training label encoding order) ──
GENDER_MAP      = {"female": 0, "male": 1}
RACE_MAP        = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
EDUCATION_MAP   = {
    "associate's degree": 0, "bachelor's degree": 1,
    "high school": 2, "master's degree": 3,
    "some college": 4, "some high school": 5
}
LUNCH_MAP       = {"free/reduced": 0, "standard": 1}
PREP_MAP        = {"completed": 0, "none": 1}


def build_features(data: dict) -> np.ndarray:
    math    = float(data["math_score"])
    reading = float(data["reading_score"])
    writing = float(data["writing_score"])

    features = {
        "gender_enc":              GENDER_MAP.get(data["gender"], 0),
        "race_ethnicity_enc":      RACE_MAP.get(data["race_ethnicity"], 0),
        "parental_education_enc":  EDUCATION_MAP.get(data["parental_education"], 0),
        "lunch_enc":               LUNCH_MAP.get(data["lunch"], 1),
        "test_preparation_enc":    PREP_MAP.get(data["test_preparation"], 1),
        "math_score":              math,
        "reading_score":           reading,
        "writing_score":           writing,
        "score_total":             math + reading + writing,
        "score_spread":            round(np.std([math, reading, writing]), 2),
        "verbal_avg":              round((reading + writing) / 2, 2),
    }
    return np.array([[features[col] for col in feature_cols]])


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        X    = build_features(data)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        return jsonify({
            "prediction":   "Pass" if pred == 1 else "Fail",
            "confidence":   round(float(max(prob)) * 100, 2),
            "pass_prob":    round(float(prob[1]) * 100, 2),
            "fail_prob":    round(float(prob[0]) * 100, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
