# Turkish Hate Speech Detection on Tweets

This project aims to develop a **hate speech detection system** for Turkish tweets using various **machine learning algorithms** and **text representation techniques**. It handles class imbalance using oversampling, undersampling, and combined resampling methods.

---

## **Project Overview**

The project involves:
1. **Preprocessing** a unified Turkish tweet dataset.
2. **Text Representation**:
   - CountVectorizer (Unigram, Bigram)
   - TfidfVectorizer (Unigram, Bigram)
3. **Classification Algorithms**:
   - Random Forest (Ensemble Method)
   - XGBoost and LightGBM (Boosting Methods)
   - Artificial Neural Network (ANN)
4. **Resampling Techniques** to address class imbalance:
   - SMOTE (Oversampling)
   - Random Undersampling
   - SMOTETomek (Combined Method)
5. **Performance Evaluation**:
   - Precision, Recall, F1-score
   - Confusion Matrix
   - Training/Validation Loss and Accuracy Visualization

---

## **Technologies Used**

The project is implemented using the following libraries and frameworks:

- **Python** 3.8+
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning models and text processing
- **imbalanced-learn**: Resampling methods for class imbalance
- **xgboost** and **lightgbm**: Boosting algorithms
- **Keras** with **TensorFlow** backend: Artificial Neural Network (ANN)
  
---

## **Installation**

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yusufcelikkirann/turkish-hate-speech.git
   cd turkish-hate-speech

## **Results**

  ### 1. Random Forest Model Results

| Class       | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| 0           | 0.78      | 1.00    | 0.87     | 1527    |
| 1           | 0.93      | 0.16    | 0.27     | 490     |
| 2           | 0.00      | 0.00    | 0.00     | 28      |
| **Macro Avg** | 0.57    | 0.39    | 0.38     | 2045    |
| **Weighted Avg** | 0.80 | 0.78    | 0.72     | 2045    |
| **Accuracy** | **0.78** | -       | -        | -       |

---

### 2. XGBoost Model Results

| Class       | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| 0           | 0.83      | 0.98    | 0.90     | 1527    |
| 1           | 0.84      | 0.39    | 0.53     | 490     |
| 2           | 0.67      | 0.29    | 0.40     | 28      |
| **Macro Avg** | 0.78    | 0.55    | 0.61     | 2045    |
| **Weighted Avg** | 0.83 | 0.83    | 0.80     | 2045    |
| **Accuracy** | **0.83** | -       | -        | -       |

---

### 3. LightGBM Model Results

| Class       | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| 0           | 0.84      | 0.97    | 0.90     | 1527    |
| 1           | 0.79      | 0.43    | 0.55     | 490     |
| 2           | 0.89      | 0.29    | 0.43     | 28      |
| **Macro Avg** | 0.84    | 0.56    | 0.63     | 2045    |
| **Weighted Avg** | 0.82 | 0.83    | 0.81     | 2045    |
| **Accuracy** | **0.83** | -       | -        | -       |

---

### 4. ANN Model Results

| Class       | Precision | Recall  | F1-Score | Support |
|-------------|-----------|---------|----------|---------|
| 0           | 0.90      | 0.89    | 0.89     | 1527    |
| 1           | 0.65      | 0.71    | 0.68     | 490     |
| 2           | 0.00      | 0.00    | 0.00     | 28      |
| **Macro Avg** | 0.52    | 0.53    | 0.52     | 2045    |
| **Weighted Avg** | 0.83 | 0.83    | 0.83     | 2045    |
| **Accuracy** | **0.83** | -       | -        | -       |


