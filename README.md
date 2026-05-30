# 🎓 Student Performance Predictor

A complete end-to-end **Machine Learning project** that predicts whether a student will **Pass or Fail** based on demographic information and academic scores.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Domain** | Education / Machine Learning |
| **Task** | Binary Classification (Pass / Fail) |
| **Best Model** | Gradient Boosting / Random Forest |
| **Tech Stack** | Python, Scikit-learn, Pandas, Flask, Power BI |

---

## 🗂️ Project Structure

```
student-performance-predictor/
│
├── data/
│   └── students.csv            # Dataset (auto-generated)
│
├── src/
│   ├── train.py                # EDA + Feature Engineering + Model Training
│   └── app.py                  # Flask REST API
│
├── models/
│   ├── best_model.pkl          # Trained model (after running train.py)
│   ├── scaler.pkl              # StandardScaler
│   └── feature_cols.pkl        # Feature column order
│
├── templates/
│   └── index.html              # Web UI for predictions
│
├── plots/
│   ├── eda_dashboard.png       # EDA visualizations
│   └── feature_importance.png  # Feature importance chart
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Hemant-sahu/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```
This will:
- Generate the dataset (`data/students.csv`)
- Run EDA and save plots
- Train 3 ML models (Logistic Regression, Random Forest, Gradient Boosting)
- Save the best model to `models/`

### 4. Start the Flask API
```bash
python src/app.py
```
Open your browser → `http://localhost:5000`

---

## 📊 Models Compared

| Model | Accuracy | CV Score |
|-------|----------|----------|
| Logistic Regression | ~87% | ~86% |
| Random Forest | ~92% | ~91% |
| **Gradient Boosting** | **~93%** | **~92%** |

---

## 🔌 API Endpoints

### POST `/predict`
```json
{
  "gender": "male",
  "race_ethnicity": "group C",
  "parental_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation": "completed",
  "math_score": 75,
  "reading_score": 80,
  "writing_score": 78
}
```

**Response:**
```json
{
  "prediction": "Pass",
  "confidence": 96.4,
  "pass_prob": 96.4,
  "fail_prob": 3.6
}
```

### GET `/health`
Returns model status.

---

## 🧠 Key Features

- **EDA Dashboard** – Score distributions, Pass/Fail split, correlation heatmap
- **Feature Engineering** – Score total, verbal average, score spread
- **3 Model Comparison** – With cross-validation
- **REST API** – Flask-based prediction endpoint
- **Web UI** – Clean interface to test predictions

---

## 👨‍💻 Author

**Hemant Sahu**  
Python Developer | Data Analyst | ML Engineer  
📧 hs948316@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/hemant-sahu) | [GitHub](https://github.com/Hemant-sahu)

---

## 📄 License

MIT License — feel free to use and modify.
