
# Sentiment Classification using Traditional ML and Deep Learning (RNN & LSTM)

## 📌 Project Overview

This project aims to build and compare sentiment classification models using both traditional machine learning techniques (e.g., Logistic Regression, SVM, Random Forest) and deep learning approaches (RNN and LSTM networks). It involves preprocessing text data, feature extraction, model training, evaluation, and performance comparison. The goal is to determine which method yields the most accurate and generalizable results for text-based sentiment analysis.

---

## 📂 Project Structure

- `final_result_with_rnn_lstm.ipynb`: Main notebook containing data loading, preprocessing, model building, training, and evaluation.
- (Optional) `data.csv`: Source dataset containing labeled text samples (assumed to be sentiment-based).
- Saved models and tokenizers (optional, if you use `pickle` or `model.save()`).

---

## 🧪 Workflow Summary

### 1. **Data Loading and Preprocessing**
- Load data using `pandas`.
- Clean and normalize text.
- Tokenization and padding for deep learning models.
- TF-IDF vectorization for classical ML models.
- Label encoding of target classes.

### 2. **Modeling Techniques**
- **Traditional ML Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest

- **Deep Learning Models**:
  - Simple RNN with Embedding Layer
  - LSTM with Embedding Layer

### 3. **Training and Evaluation**
- Metrics used: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix for visual analysis
- Comparison of performance between traditional and deep models.

---

## 📊 Key Features

- Deep learning implemented using Keras (TensorFlow backend).
- Efficient preprocessing using `nltk` and `sklearn`.
- Evaluation via standard metrics.
- Use of `transformers` library for additional sentiment baselines.

---

## 🛠️ Installation and Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk transformers
```

Make sure you also download NLTK lexicons:

```python
import nltk
nltk.download('vader_lexicon')
```

---

## ▶️ How to Run

1. Open the notebook:
   ```bash
   jupyter notebook final_result_with_rnn_lstm.ipynb
   ```
2. Run all cells sequentially.
3. Review performance metrics and charts at the end.

---

## ✅ Results Summary


- Best model: Roberta.
- Accuracy achieved: 85%
---

## 📌 Dependencies

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- TensorFlow (Keras)
- NLTK
- Transformers (HuggingFace)

---

## 📧 Contact

For any questions or issues, please contact the me 
linkedin : https://www.linkedin.com/in/mahmoud-amr-b557bb249/.

