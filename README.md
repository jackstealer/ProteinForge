# Agricultural Crop Yield Prediction

## Machine Learning-Based Yield Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Problem Statement

Predicting agricultural crop yields is crucial for food security, economic planning, and resource management. This project develops machine learning models to accurately predict crop yields based on environmental factors including:

- Rainfall patterns
- Temperature conditions
- Pesticide usage
- Geographical location
- Crop type

Accurate yield prediction enables farmers, policymakers, and agricultural stakeholders to make informed decisions about crop planning, resource allocation, and market strategies.

---

## 📊 Dataset Description

The agricultural dataset contains comprehensive information about crop production across different regions and time periods:

### Dataset Characteristics:

- **Total Records**: 28,242 initial records (25,932 after preprocessing)
- **Features**: 7 input features + 1 target variable
- **Data Quality**: No missing values
- **Duplicates Removed**: 2,310 duplicate records (8.2% of original data)

### Features:

| Feature                         | Description                                   | Type        |
| ------------------------------- | --------------------------------------------- | ----------- |
| `Area`                          | Geographic region/country                     | Categorical |
| `Item`                          | Crop type (Maize, Potatoes, Rice, etc.)       | Categorical |
| `Year`                          | Production year                               | Numerical   |
| `hg/ha_yield`                   | Crop yield in hectograms per hectare (TARGET) | Numerical   |
| `average_rain_fall_mm_per_year` | Annual rainfall in millimeters                | Numerical   |
| `pesticides_tonnes`             | Pesticide usage in tonnes                     | Numerical   |
| `avg_temp`                      | Average temperature                           | Numerical   |

### Data Sources:

- `pesticides.csv` - Pesticide usage data
- `rainfall.csv` - Rainfall measurements
- `temp.csv` - Temperature records
- `yield.csv` - Crop yield data
- `yield_df.csv` - Merged dataset (main data file)

---

## 🛠️ Methodology

### 1. Data Preprocessing

- **Missing Value Treatment**: Verified no missing values
- **Duplicate Removal**: Removed 2,310 duplicate records
- **Feature Engineering**:
  - Label encoding for categorical variables (Area, Item)
  - Standard scaling for numerical features
- **Data Split**: 80% training, 20% testing

### 2. Exploratory Data Analysis (EDA)

- Distribution analysis of crop yields
- Correlation analysis between features
- Scatter plots: Rainfall vs Yield, Temperature vs Yield, Pesticides vs Yield
- Top 10 crop frequency analysis
- Feature correlation heatmap

### 3. Models Implemented

#### a) Linear Regression

- **Type**: Baseline linear model
- **Purpose**: Comparison benchmark
- **Use Case**: Simple linear relationships

#### b) Random Forest Regressor

- **Type**: Ensemble learning (Decision Trees)
- **Parameters**:
  - n_estimators=200
  - random_state=42
  - n_jobs=-1 (parallel processing)
- **Advantages**: Handles non-linearity, feature interactions

#### c) XGBoost Regressor

- **Type**: Gradient boosting algorithm
- **Parameters**:
  - n_estimators=300
  - learning_rate=0.05
  - max_depth=8
  - random_state=42
- **Advantages**: High performance, complex pattern recognition

### 4. Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute difference
- **MSE** (Mean Squared Error): Squared error penalty
- **RMSE** (Root Mean Squared Error): Standard deviation of errors
- **R² Score**: Proportion of variance explained (0-1 scale)

---

## 📈 Results

### Model Performance Comparison

| Model                | MAE          | MSE             | RMSE          | R² Score  | Accuracy  |
| -------------------- | ------------ | --------------- | ------------- | --------- | --------- |
| **Random Forest** ⭐ | **4,216.10** | **121,520,816** | **11,023.65** | **0.983** | **98.3%** |
| XGBoost              | 6,216.13     | 163,386,944     | 12,782.29     | 0.977     | 97.7%     |
| Linear Regression    | 62,177.71    | 6,668,570,972   | 81,661.32     | 0.080     | 8.0%      |

### Feature Importance (Random Forest)

| Rank | Feature              | Importance | Impact                         |
| ---- | -------------------- | ---------- | ------------------------------ |
| 1    | **Item (Crop Type)** | 59.9%      | Most critical factor           |
| 2    | Pesticides Usage     | 11.8%      | Significant input factor       |
| 3    | Average Temperature  | 10.3%      | Important environmental factor |
| 4    | Rainfall             | 7.9%       | Moderate environmental impact  |
| 5    | Area (Geography)     | 6.9%       | Regional influence             |
| 6    | Year                 | 3.1%       | Temporal trends                |

### Key Insights:

✅ **Crop type** is the dominant factor (60% importance)  
✅ **Agricultural inputs** (pesticides) significantly impact productivity  
✅ **Climate factors** (temperature, rainfall) contribute ~18%  
✅ **Random Forest** outperformed other models by significant margin

---

## 🎯 Conclusion

### Project Success Highlights:

- ✅ Achieved **98.3% prediction accuracy** with Random Forest
- ✅ Identified **crop type as primary yield driver** (60% importance)
- ✅ Demonstrated significant improvement over linear approaches
- ✅ Clean data pipeline with robust preprocessing

### Business Impact:

- **Farmers**: Optimize crop selection and input usage
- **Policymakers**: Plan agricultural strategies and food security measures
- **Researchers**: Understand factors affecting agricultural productivity
- **Markets**: Forecast supply and demand patterns

### Technical Excellence:

Strong data science methodology with:

- Comprehensive data preprocessing
- Multiple model comparison
- Thorough evaluation metrics
- Actionable insights for stakeholders

---

## 📁 Project Structure

```
INTERN/
│
├── data/                          # Dataset files
│   ├── pesticides.csv
│   ├── rainfall.csv
│   ├── temp.csv
│   ├── yield.csv
│   └── yield_df.csv
│
├── model/                         # Trained models (.pkl files)
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── linear_regression_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder_area.pkl
│   └── label_encoder_item.pkl
│
├── notebook/                      # Jupyter notebooks
│   └── agricultural_yield_prediction.ipynb
│
├── results/                       # Visualizations & metrics
│   ├── yield_distribution.png
│   ├── rainfall_vs_yield.png
│   ├── temperature_vs_yield.png
│   ├── pesticides_vs_yield.png
│   ├── top_10_crops.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── model_comparison.csv
│   └── feature_importance.csv
│
├── README.md                      # Project documentation (this file)
├── Agricultural_Yield_Prediction_Report.md  # Detailed report
├── Agricultural_Yield_Prediction_Report.pdf # PDF report
└── new.py                         # Python script version
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/jackstealer/INTERNSHIP.git
cd INTERNSHIP
```

2. **Install required packages**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

3. **Run the Jupyter Notebook**

```bash
jupyter notebook notebook/agricultural_yield_prediction.ipynb
```

4. **Or run the Python script**

```bash
python new.py
```

---

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
```

---

## 📊 Visualizations

The project generates multiple visualizations saved in the `results/` folder:

- Yield distribution histogram
- Scatter plots (Rainfall, Temperature, Pesticides vs Yield)
- Top 10 crops bar chart
- Correlation heatmap
- Model comparison chart
- Feature importance chart
- Actual vs Predicted scatter plot

---

## 🔮 Future Enhancements

1. **Model Optimization**: Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
2. **Deep Learning**: Implement neural network models (LSTM, CNN)
3. **Feature Engineering**: Create interaction features and polynomial terms
4. **Time Series Analysis**: Develop seasonal prediction models
5. **Regional Models**: Build location-specific prediction models
6. **Real-time Integration**: Deploy API for continuous yield monitoring
7. **Web Dashboard**: Create interactive visualization dashboard
8. **Mobile App**: Develop farmer-facing mobile application

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@jackstealer](https://github.com/jackstealer)
- Repository: [INTERNSHIP](https://github.com/jackstealer/INTERNSHIP)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset sources for agricultural data
- Open-source community for ML libraries
- Contributors and reviewers

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub or contact through the repository.

---

**⭐ If you find this project useful, please consider giving it a star!**

---

_Last Updated: June 27, 2026_
