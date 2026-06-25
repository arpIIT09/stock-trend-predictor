# 📈 Stock Trend Predictor — SOC Mid-Term Submission

**Mentors:** Ramagiri Harshitha (24b0053) | **Co-Mentor:** Lacky Yadav (24b0026)  
**Category:** Machine Learning  
**Goal:** Predict whether a stock's price will go **Up** or **Down** the next trading day using historical data and machine learning.

---

## 📁 Repository Structure

```
stock-trend-predictor/
├── notebooks/
│   ├── Week1_Data_Collection.ipynb
│   ├── Week2_EDA_and_Visualization.ipynb
│   ├── Week3_Feature_Engineering.ipynb
│   └── Week4_ML_Models.ipynb
├── data/
│   ├── raw/
│   │   └── AAPL_raw.csv          ← downloaded via yfinance
│   └── processed/
│       ├── AAPL_clean.csv        ← after cleaning (Week 2)
│       ├── AAPL_features.csv     ← ML-ready dataset (Week 3)
│       └── README.md
├── plots/                        ← saved visualizations
└── README.md
```

> **Run order:** Week1 → Week2 → Week3 → Week4. Each notebook depends on files saved by the previous one.

---

## 📅 Weekly Progress

### Week 1 — Python Basics & Data Collection

**Objective:** Set up the environment and collect real stock market data.

**Work done:**
- Installed and configured `yfinance`, `pandas`, and `matplotlib`
- Downloaded 2 years of AAPL historical data (Open, High, Low, Close, Volume)
- Explored the dataset shape, column types, and basic statistics
- Verified no missing values in the initial raw data
- Saved raw data to `data/raw/AAPL_raw.csv`

**Key Observations:**
- Stock data comes in OHLCV format — 5 standard columns representing daily price activity
- The dataset covers ~500 trading days over 2 years
- `yfinance` occasionally returns MultiIndex columns that need to be flattened

---

### Week 2 — Data Cleaning, EDA & Visualization

**Objective:** Understand and visualize stock price behavior; clean the dataset for downstream use.

**Work done:**
- Performed full data cleaning: checked for nulls, handled missing values with forward-fill, removed duplicate index entries
- Computed new columns: `Daily_Range` (High − Low), `OC_Change` (% open-to-close), `Daily_Return` (% day-to-day)
- Built 20-day and 50-day moving averages (`MA20`, `MA50`)
- Created visualizations:
  - Closing price over time
  - Price + Volume subplot (identifying high-volume events)
  - Normalized multi-stock comparison (AAPL, TSLA, GOOGL)
  - Moving average overlays
  - Daily return distribution
  - Correlation heatmap of all features
- Saved cleaned data to `data/processed/AAPL_clean.csv`

**Key Observations:**
- AAPL shows a clear long-term upward trend over the 2-year period, with periods of significant volatility
- Volume spikes consistently accompany sharp price movements, confirming that volume validates price action
- Daily returns are approximately normally distributed, centered near 0%, with a slight positive skew
- AAPL, TSLA, and GOOGL show positive correlation but meaningfully different performance trajectories
- Moving average crossovers (MA20 vs MA50) visually correspond to medium-term trend changes

---

### Week 3 — Feature Engineering & Target Creation

**Objective:** Transform raw stock data into a structured ML-ready dataset with 8 meaningful features and a binary target label.

**Work done:**
- Loaded the cleaned Week 2 CSV and rebuilt base columns
- Engineered **8 features** from scratch:

| Feature | What it captures |
|---------|-----------------|
| `MA5` | 5-day moving average — very short-term trend |
| `MA5_to_MA20` | Ratio of short-term to long-term MA (Golden/Death Cross signal) |
| `Price_to_MA20` | Is today's price above its recent average? |
| `Daily_Return` | Today's % price change |
| `Lag1_Return` | Yesterday's return (one-day memory) |
| `Lag2_Return` | Two days ago's return |
| `Volatility_5d` | 5-day rolling standard deviation of returns |
| `Volume_Change` | % change in trading volume vs yesterday |

- Created the **Target column**: `1` if tomorrow's close > today's close, `0` otherwise
- Verified no data leakage (used `shift(-1)` for target, `shift(+1)` for lag features)
- Dropped NaN rows introduced by rolling windows and lag shifts
- Saved ML-ready dataset to `data/processed/AAPL_features.csv`

**Key Observations:**
- Target distribution is approximately 53% Up / 47% Down — a healthy, near-balanced split. No class imbalance problem.
- `MA5_to_MA20` ratio oscillates around 1.0 and visually tracks medium-term momentum well
- `Volatility_5d` spikes correspond precisely to periods of sharp price swings in the raw chart
- Data leakage is the #1 beginner mistake: `shift(-1)` on the target is safe (we're looking forward), but any feature that uses future prices would corrupt training

---

### Week 4 — Machine Learning Models

**Objective:** Build, train, and evaluate two classification models on the engineered feature set.

**Work done:**
- Loaded `AAPL_features.csv` and defined `X` (8 features) and `y` (Target)
- Applied a **time-series-safe 80/20 train-test split** (no shuffling — trained on earlier dates, tested on later dates)
- Trained **Logistic Regression** (baseline model):
  - Learns a weighted linear combination of features
  - Fast, interpretable, and a reliable baseline
- Trained **Decision Tree Classifier** (depth-limited to 4):
  - Learns a series of threshold-based yes/no rules
  - Visualized the tree to inspect learned rules
- Evaluated both models with:
  - **Accuracy** on train and test sets
  - **Confusion Matrix** to understand Up vs Down prediction quality
  - **Classification Report** (Precision, Recall, F1-score)
- Studied the **overfitting vs underfitting** tradeoff by varying `max_depth`

**Results:**

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|--------------|
| Logistic Regression | ~52–54% | ~52–55% |
| Decision Tree (depth=4) | ~58–62% | ~52–55% |

**Key Observations:**
- Both models achieve test accuracy in the 52–55% range, which is meaningful given that random guessing gives 50%
- Logistic Regression has very similar train/test accuracy — low variance, no overfitting
- Decision Tree shows a gap between train and test accuracy — a mild overfitting signal that worsens as `max_depth` increases
- Unconstrained Decision Trees (no `max_depth`) overfit severely: ~100% train accuracy, ~50% test accuracy — exactly the memorization problem described in the lesson
- The confusion matrices reveal that both models tend to predict "Up" more often — class 1 has slightly higher recall

---

## 🔑 Custom Features & Models

- **`MA5_to_MA20` ratio** — instead of feeding raw MA5 and MA20 separately, this derived ratio directly encodes the Golden Cross / Death Cross signal traders use, reducing redundancy and making the relationship explicit to the model.
- **`Lag1_Return` and `Lag2_Return`** — give the model short-term memory without any future data leakage.
- **`Volatility_5d`** — a market regime feature: the model can learn to behave differently in volatile vs calm periods.
- **Careful leakage prevention** — `shift(-1)` for target creation, `shift(+1)` for lag features, and time-ordered train/test split rather than random shuffling.

---

## 📊 Results & Conclusions

**Conclusions from 4 weeks of work:**

1. **Data quality matters most.** The cleaning and feature engineering steps (Weeks 2–3) took the most effort and have the biggest impact on model performance.

2. **Stock price direction prediction is genuinely hard.** Markets are highly efficient; a 52–55% test accuracy is a real, meaningful result — not a sign of a broken model.

3. **Simpler models are surprisingly competitive.** Logistic Regression matches or beats a Decision Tree on test data despite being far simpler. This is a consistent pattern in financial ML.

4. **Overfitting is the main practical risk.** Unconstrained models achieve near-perfect training accuracy but fail completely on held-out data. Regularization (`max_iter`, `max_depth`) is essential.

5. **The foundation is ready for improvement.** Week 5 will introduce Random Forests and better feature selection; the current ML-ready dataset is a solid base.

---

## ⚙️ Setup & Running

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd stock-trend-predictor

# 2. Install dependencies
pip install yfinance pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Run notebooks in order
jupyter notebook notebooks/Week1_Data_Collection.ipynb
jupyter notebook notebooks/Week2_EDA_and_Visualization.ipynb
jupyter notebook notebooks/Week3_Feature_Engineering.ipynb
jupyter notebook notebooks/Week4_ML_Models.ipynb
```

**Python version:** 3.8+  
**Key libraries:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 📚 References

- [yfinance documentation](https://pypi.org/project/yfinance/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle Pandas micro-course](https://www.kaggle.com/learn/pandas)
- [StatQuest — Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [StatQuest — Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y)
