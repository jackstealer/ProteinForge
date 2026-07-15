# 🚀 Thiranex Internship — Data Science & Machine Learning Projects

> A collection of **4 hands-on Data Science and Machine Learning tasks** completed as part of the Thiranex internship program. Each task progressively builds expertise in data cleaning, predictive modeling, exploratory analysis, and real-world AI application.

---

## 👤 Intern Details

| Field        | Details                        |
|--------------|-------------------------------|
| **Name**     | Atul Raj Gautam               |
| **Program**  | Thiranex Data Science Internship |
| **GitHub**   | [sololeveling0415-hue](https://github.com/sololeveling0415-hue) |
| **Repo**     | [Thiranex_internship](https://github.com/sololeveling0415-hue/Thiranex_internship) |

---

## 📁 Repository Structure

```
Thiranex_internship/
├── README.md                        ← You are here
├── task1/                           ← Data Cleaning & Visualization
│   ├── data_cleaning_visualization.py
│   ├── movies_cleaned.csv
│   ├── requirements.txt
│   └── visualizations/
├── task2/                           ← Predictive Modeling (ML)
│   ├── predictive_modeling.py
│   ├── healthcare-dataset-stroke-data.csv
│   ├── model_report.txt
│   ├── requirements.txt
│   └── visualizations/
├── task3/                           ← Exploratory Data Analysis
│   ├── exploratory_data_analysis.py
│   ├── Global YouTube Statistics.csv
│   ├── eda_insights_report.txt
│   ├── requirements.txt
│   └── visualizations/
└── task4/                           ← Real-World ML Project (Tic-Tac-Toe)
    ├── tic_tac_toe_analysis.py
    ├── play_tic_tac_toe.py
    ├── tic_tac_toc.csv
    ├── requirements.txt
    └── visualizations/
```

---

## 📌 Task Summaries

---

### ✅ Task 1 — Data Cleaning & Visualization

> **Domain:** Data Preprocessing & Visual Storytelling
> **Dataset:** Movies Dataset (`movies.csv` → cleaned to `movies_cleaned.csv`)
> **Script:** `task1/data_cleaning_visualization.py`

#### 🎯 Objective
Clean and preprocess a raw movies dataset, handle data quality issues, and create compelling visualizations that tell a meaningful story about movie trends and ratings.

#### 🔧 What Was Done

| Step | Description |
|------|-------------|
| **Missing Values** | Imputed using median (numeric) and placeholder strings (text) |
| **Duplicate Removal** | Detected and removed exact duplicate records |
| **Outlier Handling** | Applied IQR method to cap extreme values |
| **Feature Engineering** | Parsed release dates, extracted year/month, created rating categories |
| **Data Type Conversion** | Ensured correct dtypes across all columns |

#### 📊 Visualizations Produced (8 Charts)

1. `1_vote_average_distribution.png` — Histogram + Boxplot of movie ratings
2. `2_rating_categories.png` — Pie chart of quality distribution
3. `3_movies_per_year.png` — Bar chart of yearly release trends
4. `4_popularity_vs_rating.png` — Scatter plot colored by vote count
5. `5_top_rated_movies.png` — Horizontal bar chart of best movies
6. `6_monthly_releases.png` — Line chart of seasonal release patterns
7. `7_vote_count_distribution.png` — Histogram + Violin plot
8. `8_correlation_heatmap.png` — Feature correlation matrix

#### 🛠️ Tech Stack
`Python` · `Pandas` · `Matplotlib` · `Seaborn` · `NumPy`

#### 📈 Key Learnings
- Real-world data cleaning pipeline
- Statistical imputation strategies
- Professional chart design with Matplotlib/Seaborn
- Data storytelling and pattern extraction

---

### ✅ Task 2 — Predictive Modeling Using Machine Learning

> **Domain:** Healthcare / Binary Classification
> **Dataset:** `healthcare-dataset-stroke-data.csv` (5,110 patient records)
> **Script:** `task2/predictive_modeling.py`
> **Report:** `task2/model_report.txt`

#### 🎯 Objective
Build and compare machine learning models to predict the likelihood of a stroke in patients based on health and demographic features.

#### 🤖 Models Implemented

| Model | Accuracy | ROC-AUC | Cross-Val |
|-------|----------|---------|-----------|
| **Logistic Regression** ⭐ | 95.21% | **0.8419** | 0.9513 |
| Random Forest | 94.91% | 0.7998 | 0.9506 |
| Gradient Boosting | 94.91% | 0.8360 | 0.9503 |
| Decision Tree | 94.81% | 0.8220 | 0.9472 |

> 🏆 **Best Model:** Logistic Regression (highest ROC-AUC of 0.8419)

#### 📊 Dataset Details

- **Total Samples:** 5,110 (4,088 train / 1,022 test)
- **Features:** Age, Gender, Hypertension, Heart Disease, Marital Status, Work Type, Residence Type, Avg Glucose Level, BMI, Smoking Status
- **Target:** `stroke` (1 = stroke, 0 = no stroke)
- **Class Imbalance:** 95.1% no stroke vs 4.9% stroke cases

#### 📊 Visualizations Produced (6 Charts)

1. `1_model_performance_comparison.png` — Side-by-side metric bars
2. `2_confusion_matrices.png` — 2×2 grid for all models
3. `3_roc_curves.png` — ROC curves per model
4. `4_precision_recall_curves.png` — PR curves
5. `5_cross_validation_scores.png` — 5-Fold CV comparison
6. `6_feature_importance.png` — Top 15 features from Random Forest

#### 💡 Key Insights
- Dataset has significant class imbalance (95.1% negative cases)
- Logistic Regression achieved the best ROC-AUC despite simpler architecture
- Low recall (0.02) across models highlights imbalance issue → SMOTE recommended
- All models achieved ~95% accuracy but metrics like F1 reveal deeper challenges

#### 🛠️ Tech Stack
`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

### ✅ Task 3 — Exploratory Data Analysis (EDA)

> **Domain:** Social Media / Content Analytics
> **Dataset:** `Global YouTube Statistics.csv` (995 top YouTube channels, 28 features)
> **Script:** `task3/exploratory_data_analysis.py`
> **Report:** `task3/eda_insights_report.txt`

#### 🎯 Objective
Conduct a comprehensive EDA on Global YouTube Statistics to uncover trends, patterns, correlations, and actionable insights about the YouTube ecosystem.

#### 🔍 Analysis Performed

| Analysis Type | Details |
|---------------|---------|
| **Univariate** | Distribution analysis, summary stats, frequency counts, outlier detection |
| **Bivariate** | Scatter plots, correlation heatmaps, paired variable relationships |
| **Data Quality** | Missing values (1,616 total), zero duplicates found |

#### 📊 Key Findings

| Metric | Value |
|--------|-------|
| Total YouTubers Analyzed | 995 |
| Average Subscribers | 22,982,412 |
| Median Subscribers | 17,700,000 |
| Top Channel | T-Series (245M subscribers) |
| Most Popular Category | Entertainment (241 channels) |
| Top Country | United States (313 channels) |
| Countries Represented | 49 |
| Avg Uploads per Channel | 9,187 |
| Most Prolific Channel | ABP NEWS (301,308 uploads) |
| Subscriber-View Correlation | **0.75** (strong positive) |

#### 📊 Visualizations Produced (10 Charts)

1. `1_subscribers_distribution.png` — Histogram + Boxplot
2. `2_top_20_youtubers.png` — Horizontal bar chart
3. `3_category_distribution.png` — Pie chart by category
4. `4_country_distribution.png` — Bar chart by country
5. `5_subscribers_vs_views.png` — Scatter plot
6. `6_correlation_heatmap.png` — Feature correlation matrix
7. `7_avg_subscribers_category.png` — Avg subscribers per category
8. `8_uploads_distribution.png` — Upload count histogram
9. `9_channel_type_distribution.png` — Channel type bar chart
10. `10_creation_timeline.png` — Channel creation over years

#### 💡 Key Insights
- A strong 0.75 correlation exists between subscribers and video views
- The US dominates YouTube with 31% of top channels, followed by India (16.9%)
- Entertainment is the largest category but "Shows" channels have the highest avg subscribers
- Massive right-skew in subscriber distribution — most channels cluster at lower counts
- Content production volume (uploads) does not strongly correlate with subscriber count

#### 🛠️ Tech Stack
`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `SciPy`

---

### ✅ Task 4 — Real-World Data Project: Tic-Tac-Toe Win Prediction + AI Game

> **Domain:** Game AI / Pattern Recognition
> **Dataset:** `tic_tac_toc.csv` (958 game board configurations)
> **Scripts:** `task4/tic_tac_toe_analysis.py` · `task4/play_tic_tac_toe.py`

#### 🎯 Objective
Apply end-to-end data science skills to a real-world game dataset — predict Tic-Tac-Toe outcomes using 7 ML algorithms, then build an **unbeatable AI** using the Minimax algorithm with Alpha-Beta pruning.

#### 🤖 ML Models Implemented (7 Algorithms)

| # | Model | Notes |
|---|-------|-------|
| 1 | Logistic Regression | Linear classification baseline |
| 2 | Decision Tree | Interpretable tree structure visualized |
| 3 | **Random Forest** ⭐ | Best performer — 100% accuracy |
| 4 | Gradient Boosting | Advanced boosting method |
| 5 | Support Vector Machine | RBF kernel |
| 6 | Naive Bayes | Probabilistic classifier |
| 7 | K-Nearest Neighbors | Instance-based learning |

> 🏆 All models achieve **>90% accuracy**. Tree-based models perform best (Random Forest: 100%).

#### 📊 Dataset Details

- **Records:** 958 complete Tic-Tac-Toe game configurations
- **Features (9):** Board positions — `top-left`, `top-middle`, `top-right`, `middle-left`, `middle-middle`, `middle-right`, `bottom-left`, `bottom-middle`, `bottom-right`
- **Values:** `x` (Player X), `o` (Player O), `b` (Blank)
- **Target:** `Class` → `positive` (X wins) or `negative` (X doesn't win)

#### 🎮 Interactive AI Game (`play_tic_tac_toe.py`)

The second script lets you play against an **unbeatable AI** powered by:

- 🧠 **Minimax Algorithm** — evaluates ALL possible game outcomes
- ⚡ **Alpha-Beta Pruning** — optimized for speed
- 🤖 **Perfect Strategy** — never makes a mistake
- 📊 **ML Enhancement** — backed by Random Forest model

```
  Positions:              Sample Game:
   1 | 2 | 3               X | _ | _
  -----------            -----------
   4 | 5 | 6               _ | O | _
  -----------            -----------
   7 | 8 | 9               _ | _ | _

Your turn (X): Enter position (1-9) or 'q' to quit
AI is analyzing all possible outcomes...
```

> ⚠️ **Warning:** You CANNOT beat this AI. Best achievable outcome = **DRAW**!

#### 📊 Visualizations Produced (8 Charts)

1. `1_class_distribution.png` — Bar + Pie of win/loss distribution
2. `2_model_comparison.png` — Horizontal bar of model accuracy
3. `3_confusion_matrices.png` — Top 4 model confusion matrices
4. `4_cross_validation.png` — 5-Fold CV with error bars
5. `5_feature_importance.png` — Board position importance rankings
6. `6_position_analysis.png` — X occupancy in winning game states
7. `7_decision_tree_structure.png` — Visual tree diagram
8. `8_metrics_heatmap.png` — All metrics across all models

#### 💡 Key Insights
- The **center square** (position 5) is most strategically critical for winning
- **Top row** completion is the strongest win indicator in the dataset
- Tree-based models achieve near-perfect accuracy due to discrete feature space
- Minimax + Alpha-Beta pruning solves Tic-Tac-Toe as a mathematically "solved game"
- Pattern recognition from ML models can inform real game-playing AI strategies

#### 🛠️ Tech Stack
`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## 🧰 Global Setup

All tasks share a similar setup process:

```bash
# Clone the repository
git clone https://github.com/sololeveling0415-hue/Thiranex_internship.git
cd Thiranex_internship

# Navigate to any task
cd task1   # or task2, task3, task4

# Install dependencies
pip install -r requirements.txt

# Run the script
python <script_name>.py
```

### Common Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

---

## 📊 Skills Acquired

| Skill Area | Topics Covered |
|------------|----------------|
| **Data Wrangling** | Missing values, duplicates, outliers, type conversion, feature engineering |
| **Visualization** | Histograms, scatter plots, heatmaps, bar charts, ROC curves, PR curves |
| **Machine Learning** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Cross-Validation, Confusion Matrices |
| **EDA** | Statistical summaries, correlation analysis, univariate & bivariate analysis |
| **Game AI** | Minimax algorithm, Alpha-Beta Pruning, perfect strategy implementation |
| **Reporting** | Automated report generation, data storytelling, insight documentation |

---

## 📅 Task Timeline

| Task | Title | Due Date | Status |
|------|-------|----------|--------|
| Task 1 | Data Cleaning & Visualization | 08 Jul 2026 | ✅ Completed |
| Task 2 | Predictive Modeling (ML) | 15 Jul 2026 | ✅ Completed |
| Task 3 | Exploratory Data Analysis | 22 Jul 2026 | ✅ Completed |
| Task 4 | Real-World ML Project (Tic-Tac-Toe) | 29 Jul 2026 | ✅ Completed |

---

## 🎓 Conclusion

These four tasks represent a complete journey through the **core data science workflow**:

1. **Clean** → raw data becomes analysis-ready
2. **Model** → patterns become predictions
3. **Explore** → data becomes insights
4. **Apply** → knowledge becomes real-world AI

Each task builds on the previous, culminating in Task 4 — a full end-to-end machine learning project combined with a playable AI game, demonstrating both analytical depth and practical application.

---

> *Created as part of the Thiranex Data Science Internship Program — 2026*
