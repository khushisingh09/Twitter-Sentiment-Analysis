# 🐦 Twitter Sentiment Analysis

An end-to-end **Twitter Sentiment Analysis Web Application** developed using Python, Natural Language Processing (NLP), and Machine Learning.

This project uses **TF-IDF vectorization and Logistic Regression** to classify tweets into Positive and Negative sentiments through an interactive Flask-based web interface.

The model is trained on the **Sentiment140 dataset**, which originally contains 1.6 million tweets.

## ✨ Key Features

- **Sentiment Classification:** Predict whether a tweet expresses positive or negative sentiment.
- **Multiple Tweet Analysis:** Enter multiple tweets simultaneously, with one tweet per line.
- **NLP-Based Text Processing:** Clean and normalize text using HTML entity handling, emoji conversion, stopword removal, and stemming.
- **TF-IDF Feature Extraction:** Convert textual data into numerical features for machine learning.
- **Logistic Regression:** Apply a supervised machine learning algorithm for binary sentiment classification.
- **Pre-trained Model Integration:** Load the trained model and vectorizer using Joblib without retraining every time.
- **Flask Web Interface:** Interact with the application through an HTML and CSS frontend.
- **Instant Predictions:** Display individual sentiment predictions for submitted tweets.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer |
| Natural Language Processing | NLTK, Porter Stemmer, Emoji |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Data Processing | Pandas |
| Model Serialization | Joblib |
| Dataset | Sentiment140 |
| Version Control | Git, GitHub |

## 📊 Dataset

This project uses the **Sentiment140 dataset**, available on Kaggle.

- **Dataset:** Sentiment140
- **Original Records:** 1,600,000 tweets
- **Original Sentiment Labels:**
  - `0` – Negative
  - `4` – Positive
- **Label Transformation:** Original labels `0` and `4` are mapped to `0` and `1`, respectively.
- **Dataset File:** `training.1600000.processed.noemoticon.csv`

**Dataset Source:** [Sentiment140 – Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

The dataset is not included in this repository because of its large file size.

## ⚙️ Project Workflow

1. **Data Collection:** Load the Sentiment140 dataset.
2. **Data Cleaning:** Remove missing records and prepare sentiment labels.
3. **Text Preprocessing:** Handle HTML entities, emojis, HTML tags, URLs, mentions, hashtags, punctuation, and special characters.
4. **Text Normalization:** Convert text to lowercase, remove stopwords, and apply Porter stemming.
5. **Train-Test Split:** Split the dataset into 80% training and 20% testing data using stratified sampling.
6. **Feature Extraction:** Apply TF-IDF vectorization with a maximum of 50,000 features.
7. **Model Training:** Train a Logistic Regression classifier with a maximum of 1,000 iterations.
8. **Model Evaluation:** Evaluate the classifier using training accuracy, testing accuracy, precision, recall, F1-score, and classification report.
9. **Model Persistence:** Save the trained classifier and TF-IDF vectorizer using Joblib.
10. **Web Integration:** Load the saved model into Flask and generate sentiment predictions for user-provided tweets.

## 📂 Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── preprocessing.py
├── trained_model.joblib
├── vectorizer.joblib
├── sample_demo.csv
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `data/` directory containing the Sentiment140 CSV file is required for training but is not included in the repository. The pre-trained model and vectorizer are included. The `result.html` template is not required by the current application.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/khushisingh09/Twitter-Sentiment-Analysis.git
```

### 2. Navigate to the Project Directory

```bash
cd Twitter-Sentiment-Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open the following URL in your browser:

```text
http://127.0.0.1:5000
```

**Note:** The pre-trained model and vectorizer files are included in the repository, so model training is not required to run the web application.

## 🧠 Model Training

To train the model from scratch:

1. Download the Sentiment140 dataset from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
2. Place the CSV file inside the expected dataset directory:

```text
data/
└── sentiment140/
    └── training.1600000.processed.noemoticon.csv
```

3. Alternatively, the training script attempts to download the dataset using `opendatasets`. Kaggle credentials and access may be required.
4. Execute:

```bash
python train.py
```

The training script performs text preprocessing, TF-IDF feature extraction, Logistic Regression training, model evaluation, and saves:

- `trained_model.joblib`
- `vectorizer.joblib`

## 📈 Model Evaluation

The Logistic Regression classifier is evaluated using an 80:20 stratified train-test split on the Sentiment140 dataset.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Training Accuracy | 78.25% |
| Testing Accuracy | 77.39% |
| Weighted Precision | 77.44% |
| Weighted Recall | 77.39% |
| Weighted F1-Score | 77.38% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.79 | 0.75 | 0.77 |
| Positive | 0.76 | 0.79 | 0.78 |

The evaluation results indicate relatively balanced performance across both sentiment classes.

## 🔮 Future Enhancements

- Introduce neutral sentiment classification.
- Add sentiment distribution visualizations.
- Improve prediction quality using advanced NLP techniques.
- Support CSV-based bulk tweet analysis.
- Compare model performance with other classification algorithms.
- Deploy the Flask application for public access.

## ⚠️ Limitations

- The current implementation supports binary sentiment classification only.
- Sarcasm, irony, and context-dependent expressions can affect prediction quality.
- Model performance depends on the training dataset and preprocessing consistency.
- Predictions should be treated as automated estimates rather than definitive interpretations of human emotions.

## 👩‍💻 Author

**Khushi Singh**

M.Tech in Data Science | Python | Machine Learning | NLP | Deep Learning

- GitHub: [@khushisingh09](https://github.com/khushisingh09)

---

⭐ If you find this project interesting, consider giving this repository a star!

**Built with Python, NLP, Machine Learning, and Flask.**