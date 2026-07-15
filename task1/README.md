# Task 1: Data Cleaning & Visualization Project

## 📋 Project Overview

This project demonstrates data preprocessing, visualization, and storytelling with data using the movies dataset. The goal is to handle missing values, outliers, and duplicates, then create meaningful visualizations to extract insights.

**Status:** ⏰ Overdue by 7 Days (Due: 08 Jul 2026)

## 🎯 Objectives

- Clean and preprocess raw movie data
- Handle missing values, outliers, and duplicates effectively
- Create compelling visualizations using Pandas, Matplotlib, and Seaborn
- Generate comprehensive reports of key findings

## 🚀 Features

### Data Cleaning

- ✅ Missing value imputation (median for numeric, placeholder for text)
- ✅ Duplicate detection and removal
- ✅ Outlier handling using IQR method
- ✅ Data type conversions and feature engineering
- ✅ Date parsing and temporal feature extraction

### Visualizations Created

1. **Vote Average Distribution** - Histogram and boxplot of movie ratings
2. **Rating Categories** - Pie chart showing distribution by quality
3. **Movies Per Year** - Bar chart of temporal release trends
4. **Popularity vs Rating** - Scatter plot with vote count coloring
5. **Top Rated Movies** - Horizontal bar chart of best movies
6. **Monthly Releases** - Line chart showing seasonal patterns
7. **Vote Count Distribution** - Histogram and violin plot
8. **Correlation Heatmap** - Feature correlation analysis

## 📊 Dataset

**Source:** `movies.csv`

**Columns:**

- `id` - Unique movie identifier
- `title` - Movie title
- `overview` - Movie description
- `release_date` - Release date
- `popularity` - Popularity score
- `vote_average` - Average rating (0-10)
- `vote_count` - Number of votes

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualizations
- **NumPy** - Numerical operations

## 📦 Installation

1. Ensure Python 3.x is installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas matplotlib seaborn numpy
```

## ▶️ How to Run

1. Navigate to the task1 directory:

```bash
cd task1
```

2. Run the main script:

```bash
python data_cleaning_visualization.py
```

3. View outputs in:
   - `movies_cleaned.csv` - Cleaned full dataset
   - `visualizations/` - All generated charts
   - `analysis_report.txt` - Comprehensive findings report

## 📈 Key Findings

The script automatically generates a detailed analysis report including:

- Rating statistics and distribution
- Popularity insights
- Top-rated movies
- Temporal trends
- Audience engagement metrics
- Data quality summary

## 🗂️ Project Structure

```
task1/
├── data_cleaning_visualization.py    # Main analysis script
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── movies_cleaned.csv                # Cleaned dataset (generated)
├── analysis_report.txt               # Analysis report (generated)
└── visualizations/                   # Visualization outputs (generated)
    ├── 1_vote_average_distribution.png
    ├── 2_rating_categories.png
    ├── 3_movies_per_year.png
    ├── 4_popularity_vs_rating.png
    ├── 5_top_rated_movies.png
    ├── 6_monthly_releases.png
    ├── 7_vote_count_distribution.png
    └── 8_correlation_heatmap.png

../movies.csv                         # Source dataset (in parent folder)
```

## 📝 Expected Outcomes

✅ Learn data preprocessing techniques
✅ Master visualization with Matplotlib and Seaborn
✅ Develop data storytelling skills
✅ Create professional data analysis reports

## 👤 Author

Created as part of the internship data analysis task series.

## 📅 Timeline

- **Assigned:** Available
- **Due Date:** 08 Jul 2026
- **Status:** Overdue by 7 Days

## 🎓 Learning Outcomes

This project teaches:

- Data quality assessment and cleaning strategies
- Handling real-world messy data
- Creating meaningful visualizations
- Statistical analysis and interpretation
- Report generation and documentation
- Python data science best practices

---

**Note:** This project is designed to build foundational skills in data preprocessing, visualization, and storytelling with data.
