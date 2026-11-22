
# train.py

from pathlib import Path

import joblib
import opendatasets as od
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import preprocess


# --------------------------------------------------
# 1. Project paths and dataset configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CSV_FILENAME = "training.1600000.processed.noemoticon.csv"

DATA_URL = "https://www.kaggle.com/datasets/kazanova/sentiment140"


# --------------------------------------------------
# 2. Locate or download dataset
# --------------------------------------------------

print("Checking dataset...")

matches = list(DATA_DIR.rglob(CSV_FILENAME)) if DATA_DIR.exists() else []

if not matches:
    print("Dataset not found. Downloading from Kaggle...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    od.download(DATA_URL, data_dir=str(DATA_DIR))

    matches = list(DATA_DIR.rglob(CSV_FILENAME))

if not matches:
    raise FileNotFoundError(
        f"{CSV_FILENAME} not found inside {DATA_DIR}. "
        "Please download the Sentiment140 CSV and place it inside the data folder."
    )

CSV_PATH = matches[0]

print("Dataset found:", CSV_PATH)


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

cols = ["target", "id", "date", "flag", "user", "text"]

print("\nLoading CSV...")

df = pd.read_csv(
    CSV_PATH,
    names=cols,
    encoding="ISO-8859-1"
)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 4. Clean dataset and map labels
# --------------------------------------------------

print("\nDropping missing values...")

df = df.dropna(subset=["text", "target"])

print("Mapping sentiment labels...")

# Sentiment140: 0 = Negative, 4 = Positive
df["target"] = df["target"].replace({4: 1})

# Keep only valid sentiment labels
df = df[df["target"].isin([0, 1])]


# --------------------------------------------------
# 5. Text preprocessing
# --------------------------------------------------

print("\nPreprocessing text...")

# Use the same preprocessing function as app.py
df["clean"] = df["text"].apply(preprocess)

# Remove empty processed tweets
df = df[df["clean"].str.strip() != ""]

print("Preprocessing completed.")
print("Final dataset shape:", df.shape)


# --------------------------------------------------
# 6. Split dataset into training and testing sets
# --------------------------------------------------

X = df["clean"].values
y = df["target"].values

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=2
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. TF-IDF feature extraction
# --------------------------------------------------

print("\nFitting TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(max_features=50000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature extraction completed.")


# --------------------------------------------------
# 8. Train Logistic Regression classifier
# --------------------------------------------------

print("\nTraining Logistic Regression classifier...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Model training completed.")


# --------------------------------------------------
# 9. Evaluate classifier
# --------------------------------------------------

print("\nEvaluating classifier...")

y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

# Training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Weighted Precision, Recall and F1-score
precision = precision_score(
    y_test, y_test_pred, average="weighted", zero_division=0
)

recall = recall_score(
    y_test, y_test_pred, average="weighted", zero_division=0
)

f1 = f1_score(
    y_test, y_test_pred, average="weighted", zero_division=0
)

print("\n========== MODEL EVALUATION RESULTS ==========")

print(f"Training Accuracy : {train_accuracy:.4f}")
print(f"Testing Accuracy  : {test_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1-Score          : {f1:.4f}")

print("\nClassification Report (Test Data):\n")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

print("==============================================")


# --------------------------------------------------
# 10. Save trained model and vectorizer
# --------------------------------------------------

print("\nSaving model and vectorizer...")

joblib.dump(model, BASE_DIR / "trained_model.joblib")
joblib.dump(vectorizer, BASE_DIR / "vectorizer.joblib")

print("Model saved: trained_model.joblib")
print("Vectorizer saved: vectorizer.joblib")


# --------------------------------------------------
# 11. Save sample demo file
# --------------------------------------------------

sample = pd.DataFrame({
    "text": [
        "I love this!",
        "I hate this product."
    ],
    "label": [1, 0]
})

sample.to_csv(BASE_DIR / "sample_demo.csv", index=False)

print("\nSample demo file saved.")
print("\nTraining and evaluation process completed successfully!")
