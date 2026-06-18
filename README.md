# Spam Detection

A machine learning project that classifies SMS/text messages as **Spam** or **Ham** (not spam) using TF-IDF vectorization and a Naive Bayes classifier.

## Overview

This project builds an end-to-end spam detection pipeline:

1. Load a labeled spam/ham dataset and preprocess the text (clean + tokenize).
2. Convert text into numerical vectors using **TF-IDF**.
3. Train a **Multinomial Naive Bayes** classifier and evaluate it using Precision, Recall, and F1-score.
4. Save the trained pipeline (vectorizer + model) so it can be reused for predictions on new, unseen messages — without retraining.

## Dataset

- **Source:** SMS Spam Collection Dataset
- **Size:** 5,572 labeled messages
- **Classes:** `ham` (legitimate) and `spam`
- **File:** `dataset/spam.csv`

## Project Structure

```
spam_detection_project/
├── dataset/
│   └── spam.csv                       # Labeled SMS dataset
├── train_model.py                     # Loads data, preprocesses, trains, evaluates, saves pipeline
├── predict.py                         # Loads saved pipeline and predicts on new messages
├── spam_detector_pipeline.joblib      # Saved TF-IDF + Naive Bayes pipeline
└── requirements.txt                   # Python dependencies

Spam_Detection.ipynb                   # Interactive Jupyter Notebook version (includes a live text-box widget)
```

## Installation

```bash
pip install -r requirements.txt
```

For the notebook's interactive checker widget, also install:

```bash
pip install ipywidgets jupyter
```

## Usage

### 1. Train the model

```bash
python train_model.py
```

This will load the dataset, clean and tokenize the text, train the TF-IDF + Naive Bayes pipeline, print evaluation metrics, and save the pipeline as `spam_detector_pipeline.joblib`.

### 2. Predict on new messages

```bash
python predict.py
```

Edit the `sample_messages` list inside `predict.py` to test your own messages.

### 3. Interactive Notebook

Open `Spam_Detection.ipynb` in Jupyter and run all cells. The last cell provides a live text box and **Check** button — type any message and instantly see whether it's predicted as Spam or Ham, along with the model's confidence score.

> **Note:** Keep the `dataset/spam.csv` file in a `dataset/` folder alongside the notebook/scripts, since the code loads it from that relative path.

## How It Works

| Stage | Technique |
|---|---|
| Text Cleaning | Lowercasing, URL removal, punctuation/number removal |
| Tokenization | Word-boundary splitting |
| Stopword Removal | scikit-learn's built-in English stopword list |
| Vectorization | TF-IDF (`TfidfVectorizer`) |
| Model | Multinomial Naive Bayes (`MultinomialNB`) |
| Persistence | `joblib` (saves vectorizer + model together as one pipeline) |

## Results

| Metric | Score |
|---|---|
| Accuracy | 96.3% |
| Precision (spam) | 0.99 |
| Recall (spam) | 0.73 |
| F1-score (spam) | 0.84 |

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- joblib
- ipywidgets (for the notebook's interactive widget)

## License

This project is intended for educational purposes.
