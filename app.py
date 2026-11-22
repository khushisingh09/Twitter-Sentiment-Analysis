
from pathlib import Path

import joblib
from flask import Flask, render_template, request

from preprocessing import preprocess


# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Model file paths
MODEL_PATH = BASE_DIR / "trained_model.joblib"
VECT_PATH = BASE_DIR / "vectorizer.joblib"


# Check whether model files exist
if not MODEL_PATH.exists() or not VECT_PATH.exists():
    raise FileNotFoundError(
        "Model files not found. Please run train.py first."
    )


# Load trained model and vectorizer
print("Loading model...")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

print("Model loaded successfully!")


# Initialize Flask application
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction_results = []
    error = None

    if request.method == "POST":

        raw = request.form.get("tweet_text") or ""

        # Support multiple tweets: one tweet per line
        tweets = [
            tweet.strip()
            for tweet in raw.splitlines()
            if tweet.strip()
        ]

        if not tweets:
            error = "Please enter at least one tweet."

        else:

            # Apply the shared preprocessing function
            clean_tweets = [
                preprocess(tweet)
                for tweet in tweets
            ]

            # Avoid making predictions for empty cleaned text
            if any(not text.strip() for text in clean_tweets):

                error = (
                    "One or more tweets contain no usable text. "
                    "Please enter meaningful text."
                )

            else:

                # Convert text into TF-IDF features
                X = vectorizer.transform(clean_tweets)

                # Predict sentiment
                predictions = model.predict(X)

                for tweet, prediction in zip(tweets, predictions):

                    sentiment = (
                        "Positive"
                        if int(prediction) == 1
                        else "Negative"
                    )

                    prediction_results.append({
                        "tweet": tweet,
                        "sentiment": sentiment
                    })

    return render_template(
        "index.html",
        results=prediction_results,
        error=error
    )


if __name__ == "__main__":

    # Local development server
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
