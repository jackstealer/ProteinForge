# Model Directory

This directory contains the trained machine learning models and preprocessing objects.

## ✅ Files Included in Repository:

- `label_encoder_area.pkl` - Label encoder for geographical areas
- `label_encoder_item.pkl` - Label encoder for crop types
- `linear_regression_model.pkl` - Linear Regression model
- `scaler.pkl` - Standard scaler for feature normalization

## ⚠️ Files Generated Locally (Too Large for Git):

- `random_forest_model.pkl` (~250 MB) - **Best performing model**
- `xgboost_model.pkl` (~4 MB) - XGBoost regression model

## 🚀 How to Generate Missing Models:

Run the main script or Jupyter notebook to generate all models:

```bash
# Using Python script
python new.py

# Or using Jupyter notebook
jupyter notebook notebook/agricultural_yield_prediction.ipynb
```

This will automatically generate all model files in this directory.

## 📊 Model Performance:

| Model                | R² Score  | MAE       | File Size |
| -------------------- | --------- | --------- | --------- |
| **Random Forest** ⭐ | **0.983** | **4,216** | ~250 MB   |
| XGBoost              | 0.977     | 6,216     | ~4 MB     |
| Linear Regression    | 0.080     | 62,178    | <1 MB     |

**Best Model**: Random Forest with 98.3% accuracy

## 📝 Note:

Large model files are excluded from Git due to GitHub's 100MB file size limit. They will be automatically generated when you run the training script.
