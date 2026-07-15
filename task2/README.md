# Task 2: Predictive Modeling Using Machine Learning

## 📋 Project Overview

Build a machine learning model to predict stroke outcomes based on healthcare data. This project implements multiple ML algorithms including Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting, with comprehensive evaluation and visualization.

**Status:** ⏰ Due Today (15 Jul 2026)

## 🎯 Objectives

- Build predictive models for stroke prediction
- Apply multiple machine learning algorithms
- Train and test models for accuracy
- Visualize performance using confusion matrices and ROC curves
- Compare model performance and select the best model

## 🚀 Key Features

### Machine Learning Algorithms

- ✅ **Logistic Regression** - Linear classification baseline
- ✅ **Decision Tree** - Non-linear decision boundaries
- ✅ **Random Forest** - Ensemble method with feature importance
- ✅ **Gradient Boosting** - Advanced boosting technique

### Model Evaluation

- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ ROC-AUC Score
- ✅ 5-Fold Cross-Validation
- ✅ Confusion Matrices
- ✅ ROC Curves
- ✅ Precision-Recall Curves

### Visualizations Created

1. **Model Performance Comparison** - Bar charts of all metrics
2. **Confusion Matrices** - 2x2 grid for all models
3. **ROC Curves** - Receiver Operating Characteristic curves
4. **Precision-Recall Curves** - PR curves for all models
5. **Cross-Validation Scores** - 5-fold CV comparison
6. **Feature Importance** - Top 15 features from Random Forest

## 📊 Dataset

**Source:** `healthcare-dataset-stroke-data.csv`

**Features:**

- `age` - Age of the patient
- `gender` - Gender (Male/Female/Other)
- `hypertension` - Has hypertension (0/1)
- `heart_disease` - Has heart disease (0/1)
- `ever_married` - Marital status (Yes/No)
- `work_type` - Type of work
- `Residence_type` - Urban/Rural
- `avg_glucose_level` - Average glucose level
- `bmi` - Body mass index
- `smoking_status` - Smoking status

**Target Variable:**

- `stroke` - 1 if patient had stroke, 0 otherwise

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Visualizations

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

1. Navigate to the task2 directory:

```bash
cd task2
```

2. Run the main script:

```bash
python predictive_modeling.py
```

3. View outputs in:
   - `model_report.txt` - Detailed model evaluation
   - `visualizations/` - All performance charts

## 📈 Expected Outcomes

✅ **Gain experience in:**

- Supervised learning techniques
- Model training and evaluation
- Performance metrics interpretation
- Feature engineering and preprocessing
- Cross-validation and model selection
- Visualization of ML model performance

## 🗂️ Project Structure

```
task2/
├── predictive_modeling.py              # Main ML script
├── README.md                           # Project documentation
├── requirements.txt                    # Dependencies
├── healthcare-dataset-stroke-data.csv  # Source dataset
```

├── model_report.txt # Model evaluation (generated)
└── visualizations/ # Visualizations (generated)
├── 1_model_performance_comparison.png
├── 2_confusion_matrices.png
├── 3_roc_curves.png
├── 4_precision_recall_curves.png
├── 5_cross_validation_scores.png
└── 6_feature_importance.png

```

## 🎓 Learning Outcomes

This project teaches:
- End-to-end machine learning pipeline
- Data preprocessing for ML
- Multiple classification algorithms
- Model evaluation and comparison
- Handling imbalanced datasets
- Feature importance analysis
- ROC and PR curve interpretation
- Cross-validation techniques

## 📝 Model Performance Metrics

### Metrics Explained:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correct positive predictions / Total positive predictions
- **Recall**: Correct positive predictions / Total actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)

---

**Due Date:** 15 Jul 2026 (Due Today)
```
