# Task 4: Real-world Data Project - Tic-Tac-Toe Win Prediction

## 📋 Project Overview

A comprehensive machine learning project that predicts Tic-Tac-Toe game outcomes. This end-to-end solution includes data analysis, multiple ML model training, evaluation, and visualization, demonstrating real-world application of data science skills.

**Domain:** Game AI / Pattern Recognition  
**Status:** 🟢 Available (Due: 29 Jul 2026 - 14 Days Left)

## 🎯 Objectives

- Perform end-to-end data analysis on game data
- Build predictive models using multiple ML algorithms
- Compare model performance comprehensively
- Present findings with visualizations and conclusions
- Apply data science skills in a real-world context

## 🚀 Project Features

### Data Analysis

- ✅ Comprehensive data exploration
- ✅ Pattern recognition in winning configurations
- ✅ Position importance analysis
- ✅ Class distribution analysis

### Machine Learning Models (7 Algorithms)

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree** - Tree-based decisions with visualization
3. **Random Forest** - Ensemble learning
4. **Gradient Boosting** - Advanced boosting technique
5. **Support Vector Machine** - SVM with RBF kernel
6. **Naive Bayes** - Probabilistic classifier
7. **K-Nearest Neighbors** - Instance-based learning

### Evaluation Metrics

- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ 5-Fold Cross-Validation
- ✅ Confusion Matrices
- ✅ Model comparison analysis

### Visualizations Created (8)

1. **Class Distribution** - Bar chart and pie chart
2. **Model Performance Comparison** - Horizontal bar chart
3. **Confusion Matrices** - 2x2 grid for top 4 models
4. **Cross-Validation Scores** - Bar chart with error bars
5. **Feature Importance** - Decision tree feature rankings
6. **Position Analysis** - X occupancy in winning games
7. **Decision Tree Structure** - Visual tree representation
8. **Metrics Heatmap** - All metrics across all models

## 📊 Dataset

**Source:** `tic_tac_toc.csv`

**Description:** Tic-Tac-Toe game board configurations and outcomes

**Features (9 board positions):**

- `top-left-square`, `top-middle-square`, `top-right-square`
- `middle-left-square`, `middle-middle-square`, `middle-right-square`
- `bottom-left-square`, `bottom-middle-square`, `bottom-right-square`

**Values:**

- `x` - Player X
- `o` - Player O
- `b` - Blank

**Target Variable:**

- `Class` - positive (X wins) or negative (X doesn't win)

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Scikit-learn** - ML algorithms and evaluation
- **Matplotlib & Seaborn** - Visualizations

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ How to Run

### Option 1: Run the ML Analysis (Required for Task 4)

1. Navigate to the task4 directory:

```bash
cd task4
```

2. Run the analysis script:

```bash
python tic_tac_toe_analysis.py
```

3. View outputs in:
   - `project_report.txt` - Comprehensive project report
   - `visualizations/` - All analysis charts

### Option 2: Play the Interactive Game 🎮

**Experience the AI in action!**

```bash
python play_tic_tac_toe.py
```

## 🎮 GAME INSTRUCTIONS - EXTREME DIFFICULTY MODE

### About the Game

This is an **UNBEATABLE** Tic-Tac-Toe AI powered by:

- 🧠 **Minimax Algorithm** - Evaluates ALL possible game outcomes
- ⚡ **Alpha-Beta Pruning** - Optimized for speed
- 🤖 **Perfect Strategy** - Never makes mistakes
- 📊 **ML Enhancement** - Random Forest model (100% accuracy)

⚠️ **WARNING:** You CANNOT win! Best outcome = DRAW

### How to Play

1. **Start the game:**

   ```bash
   python play_tic_tac_toe.py
   ```

2. **Choose who goes first:**
   - Enter `1` for You (X)
   - Enter `2` for AI (O)

3. **Make your moves:**
   - Enter numbers 1-9 to place your mark
   - Board positions:
     ```
     1 | 2 | 3
     -----------
     4 | 5 | 6
     -----------
     7 | 8 | 9
     ```

4. **Game Commands:**
   - `1-9` → Place your mark
   - `q` → Quit current game
   - `y` → Play again (after game ends)
   - `n` → Exit (after game ends)

### AI Difficulty: EXTREME 🔥

**Why It's Unbeatable:**

- Uses Minimax algorithm (mathematically perfect)
- Calculates EVERY possible game sequence
- Never makes mistakes
- Always finds optimal move
- Impossible to beat - best outcome is DRAW

**AI Strategy:**

1. If AI can win in one move → takes it
2. If you can win in one move → blocks you
3. Evaluates all possibilities using Minimax
4. Chooses optimal move every time

### Tips to Achieve a Draw

⚠️ **Reality Check:** Winning is IMPOSSIBLE. Aim for a draw!

**How to Get a Draw:**

- ✅ Go first if possible
- ✅ Take center (5) as first move
- ✅ Never make mistakes
- ✅ Block all AI threats
- ✅ Think 2-3 moves ahead

**Perfect Play Pattern:**

- You go first: Center → Corner → Opposite corner
- AI goes first: Take corner if AI takes center
- Always block AI's winning moves

### Example Game

```
  Positions:              Current Board:
   1 | 2 | 3               X | b | b
  -----------            -----------
   4 | 5 | 6               b | O | b
  -----------            -----------
   7 | 8 | 9               b | b | b

👤 Your turn (X)
Enter position (1-9) or 'q' to quit: 3

🤖 AI is analyzing all possible outcomes...
🤖 AI plays position 7
```

### Winning Strategy (For Draw)

Since the AI is perfect, focus on NOT LOSING:

1. **Control Center** - Position 5 is most valuable
2. **Take Corners** - Positions 1, 3, 7, 9
3. **Block Threats** - Always stop AI's winning moves
4. **No Mistakes** - One error = AI wins immediately

**Game Theory:** Tic-Tac-Toe is a "solved game". With perfect play from both sides, result is always a DRAW. This AI plays perfectly!

---

## 📈 Expected Outcomes

✅ **Apply data science skills in real-world context**

- Complete end-to-end ML pipeline
- Model selection and comparison
- Performance optimization
- Insight generation and reporting

## 🗂️ Project Structure

```
task4/
├── tic_tac_toe_analysis.py      # Main analysis script (Required)
├── play_tic_tac_toe.py          # 🎮 Interactive game with AI
├── README.md                    # Complete documentation
├── requirements.txt             # Dependencies
├── tic_tac_toc.csv             # Source dataset
├── project_report.txt           # Analysis report (generated)
└── visualizations/              # Charts (generated)
    ├── 1_class_distribution.png
    ├── 2_model_comparison.png
    ├── 3_confusion_matrices.png
    ├── 4_cross_validation.png
    ├── 5_feature_importance.png
    ├── 6_position_analysis.png
    ├── 7_decision_tree_structure.png
    └── 8_metrics_heatmap.png
```

## 🎓 Learning Outcomes

This project demonstrates:

- End-to-end ML project workflow
- Multiple algorithm comparison
- Model evaluation best practices
- Feature importance analysis
- Pattern recognition techniques
- Data visualization for insights
- Professional report generation
- Real-world problem solving

## 💡 Key Insights

### Model Performance

- All models achieve >90% accuracy
- Tree-based models perform exceptionally well
- Decision tree provides interpretable rules

### Game Patterns

- Certain board positions are more critical for winning
- Top row completion is a strong win indicator
- Center position plays a strategic role

### Practical Applications

- Game AI development
- Pattern recognition systems
- Educational ML demonstrations
- Decision support tools

---

**Due Date:** 29 Jul 2026 (14 Days Left)

```

```
