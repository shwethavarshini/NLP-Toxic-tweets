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

## 💻 Codebase Implementation Reference

The file `toxic_tweets_classifier.py` houses the full structural execution:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# 1. Convert the CSV file to the pandas data frame
df = pd.read_csv('toxic_tweets.csv')
df['tweet'] = df['tweet'].fillna('') # Handle missing inputs gracefully

# Replace with the exact column schemas found in your downloaded CSV
X = df['tweet']
y = df['target'] 

# Preserving class imbalance layout using stratify=y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Convert text to NLP Feature Spaces
# A. Bag of Words Configuration
bow_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# B. TF-IDF Configuration
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Core Metric Extraction Module
def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name, feature_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    # Check for probability functions to guarantee smooth ROC mapping
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
        
    print(f"\n================ {model_name} with {feature_name} ================")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=['Non-Toxic', 'Toxic']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    
    auc_score = roc_auc_score(y_te, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    return fpr, tpr, auc_score

# 3. Model Orchestration Declarations
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "K-NN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM": CalibratedClassifierCV(LinearSVC(random_state=42)) # Wrapped for probability calibration
}

feature_sets = {
    "Bag of Words": (X_train_bow, X_test_bow),
    "TF-IDF": (X_train_tfidf, X_test_tfidf)
}

# 4. Pipeline Execution Loop
results = {}
for feat_name, (X_tr, X_te) in feature_sets.items():
    for model_name, model in models.items():
        fpr, tpr, auc = evaluate_model(model, X_tr, X_te, y_train, y_test, model_name, feat_name)
        results[f"{model_name} ({feat_name})"] = (fpr, tpr, auc)

# 5. Output Performance Diagnostics Matrix (ROC Curves)
plt.figure(figsize=(12, 8))
for label, (fpr, tpr, auc) in results.items():
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess Baseline")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Performance Diagnostics - Multi-Model ROC Curves Matrix')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

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

---

## 📄 License

This benchmark repository is open-sourced for educational, instructional, and research applications.
