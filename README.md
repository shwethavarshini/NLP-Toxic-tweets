# Toxic Tweets NLP Classifier

An end-to-end Natural Language Processing (NLP) pipeline designed to classify tweets into **Toxic (`1`)** or **Non-toxic (`0`)**. This project implements and benchmarks multiple text-vectorization methodologies against a collection of robust machine learning classification models.

## 📌 Project Overview

Online moderation relies heavily on parsing high-velocity, unstructured text data. This repository builds a complete comparative framework to study how different feature extraction methods (Bag of Words vs. TF-IDF) interact with distinct algorithmic paradigms—ranging from parametric estimators like Naive Bayes to high-dimensional hyperplanes like SVMs.

> **Credits:** All dataset collection credits belong to the original data creators hosted on [Kaggle](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset).

---

## 🛠️ Pipeline Architecture & Workflow

The codebase executes a modular workflow structured to avoid information leakage (ensuring text vectorizers are fit *exclusively* on training partitions):

1. **Data Ingestion:** Loads raw tweet text into a unified tabular `pandas` DataFrame, handling missing or null items seamlessly.
2. **Stratified Partitioning:** Splits data into an 80/20 train/test distribution, retaining clean class distribution ratios across partitions.
3. **Feature Space Conversions:**
* **Bag of Words (BoW):** Translates raw token occurrences into numerical frequency maps using `CountVectorizer`.
* **TF-IDF:** Normalizes raw frequencies using Term Frequency-Inverse Document Frequency weighting (`TfidfVectorizer`) to penalize ubiquitous stop words.


4. **Model Training Matrix:** Fits five independent classifiers across both feature groups.
5. **Evaluation Diagnostics:** Generates complete metric reports tracking Precision, Recall, F1-Scores, Confusion Matrices, and ROC-AUC curve mappings.

---

## 📊 Experimental Matrix Supported

| Classifier Model | Feature Vectorization Type | Evaluation Performance Captured |
| --- | --- | --- |
| **Decision Trees** | Bag of Words & TF-IDF | Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| **Random Forest** | Bag of Words & TF-IDF | Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| **Naive Bayes (Multinomial)** | Bag of Words & TF-IDF | Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| **K-Nearest Neighbors (K-NN)** | Bag of Words & TF-IDF | Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| **Support Vector Machine (SVM)** | Bag of Words & TF-IDF | Precision, Recall, F1, Confusion Matrix, ROC-AUC |

---

## 🚀 Setup & Execution Instructions

### Prerequisites

Make sure you have Python 3.8 or higher installed. Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn matplotlib

```

### Project Launch

1. Download the source dataset archive from the Kaggle Competition page.
2. Unpack and move the CSV into your working root directory, ensuring it is named `toxic_tweets.csv`.
3. Launch the comprehensive modeling script:

```bash
python toxic_tweets_classifier.py

```

---

## 📈 Understanding the Performance Diagnostics

When reviewing the runtime console execution logs, use these metrics to optimize the model deployment:

* **Precision vs. Recall Trade-Off:** High **Precision** ensures that non-toxic comments are rarely misclassified as toxic (minimizing false positives). High **Recall** ensures that malicious or toxic comments are rarely missed (minimizing false negatives).
* **Confusion Matrix Parsing:** The quadrant arrays layout indicates specific behavioral properties:
* `[0,0]` **True Negatives:** Safe tweets properly designated as safe.
* `[0,1]` **False Positives:** Flawed flag events (safe tweets flagged as toxic).
* `[1,0]` **False Negatives:** Missed violations (toxic text slipping past classification).
* `[1,1]` **True Positives:** Correctly recognized text violations.


* **ROC-AUC Curve Geometry:** Explores the true positive capability across custom classification boundaries. The closer a model's curve arcs toward the top left corner (maximizing Area Under the Curve near `1.0`), the stronger its structural predictive power.

