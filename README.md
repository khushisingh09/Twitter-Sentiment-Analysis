# 🐦 Twitter Sentiment Analysis

An end-to-end **Twitter Sentiment Analysis Web Application** developed using Python, Natural Language Processing (NLP), and Machine Learning. This project uses Logistic Regression and TF-IDF vectorization to classify tweets into Positive and Negative sentiments through an interactive Flask-based web interface.

The model is trained on the **Sentiment140 dataset**, containing 1.6 million tweets.

## ✨ Key Features

- **Sentiment Classification:** Predict whether a tweet expresses positive or negative sentiment.
- **Multiple Tweet Analysis:** Enter multiple tweets simultaneously, with one tweet per line.
- **NLP-Based Text Processing:** Clean and normalize text using tokenization, stopword removal, and stemming.
- **TF-IDF Feature Extraction:** Convert textual data into numerical features for machine learning.
- **Logistic Regression:** Apply a supervised machine learning algorithm for binary sentiment classification.
- **Pre-trained Model Integration:** Load the trained model and vectorizer using Joblib without retraining every time.
- **Flask Web Interface:** Interact with the application through a simple HTML and CSS frontend.
- **Instant Results:** Display individual sentiment predictions for submitted tweets.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer |
| Natural Language Processing | NLTK, Porter Stemmer |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Dataset | Sentiment140 |
| Version Control | Git, GitHub |

## 📊 Dataset

This project uses the **Sentiment140 dataset**, available on Kaggle.

- Dataset: Sentiment140
- Total Records: 1,600,000 tweets
- Original sentiment labels:
  - `0` – Negative
  - `4` – Positive
- Labels are transformed from `0` and `4` into `0` and `1` for binary classification.
- Dataset file: `training.1600000.processed.noemoticon.csv`

**Dataset Source:** [Sentiment140 – Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

The dataset is not included in this repository because of its large file size.

## ⚙️ Project Workflow

1. **Data Collection:** Load the Sentiment140 dataset.
2. **Data Cleaning:** Remove missing records and prepare sentiment labels.
3. **Text Preprocessing:** Handle HTML entities, emojis, HTML tags, URLs, mentions, hashtags, punctuation, and special characters.
4. **Text Normalization:** Apply lowercase conversion, stopword removal, and Porter stemming.
5. **Train-Test Split:** Split the dataset into 80% training and 20% testing data using stratified sampling.
6. **Feature Extraction:** Apply TF-IDF vectorization with a maximum of 50,000 features.
7. **Model Training:** Train a Logistic Regression classifier with a maximum of 1,000 iterations.
8. **Model Evaluation:** Evaluate predictions using accuracy score and classification report.
9. **Model Persistence:** Save the trained classifier and TF-IDF vectorizer using Joblib.
10. **Web Integration:** Load the saved model into Flask and generate sentiment predictions for user-provided tweets.

## 📂 Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── data/
│   └── sentiment140/
│       └── training.1600000.processed.noemoticon.csv
│
├── static/
│   └── styles.css
│
├── templates/
│   ├── index.html
│   └── result.html
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

*The dataset directory is required for model training but is not included in the GitHub repository. The result.html file is currently empty and can be removed if it is not used.*

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later
- pip package manager
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Twitter-Sentiment-Analysis.git
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

1. Download the Sentiment140 dataset from Kaggle.
2. Place the CSV file in the expected dataset directory.
3. Configure Kaggle credentials if using the automatic dataset download functionality.
4. Execute:

```bash
python train.py
```

The training script performs preprocessing, TF-IDF feature extraction, Logistic Regression training, model evaluation, and saves:

- `trained_model.joblib`
- `vectorizer.joblib`


## Model Evaluation

The Logistic Regression classifier is evaluated using an 80:20 train-test split on the Sentiment140 dataset.

### Evaluation Metrics

- Training Accuracy: 78.25%
- Testing Accuracy: 77.39%
- Weighted Precision: 77.44%
- Weighted Recall: 77.39%
- Weighted F1-Score: 77.38%

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.79 | 0.75 | 0.77 |
| Positive | 0.76 | 0.79 | 0.78 |

The model uses TF-IDF feature extraction and Logistic Regression for binary sentiment classification.


## 🔮 Future Enhancements

- Introduce neutral sentiment classification.
- Add sentiment distribution visualizations.
- Improve prediction quality using advanced NLP techniques.
- Support CSV-based bulk tweet analysis.
- Add model performance comparison with other classification algorithms.
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