# ⚽ FIFA World Cup 2026 — Match Predictor
### Netherlands 🇳🇱 vs Tunisia 🇹🇳 | Group F · Matchday 3 · June 25, 2026

> A full end-to-end machine learning and deep learning pipeline to predict the outcome, win/draw/loss probabilities, and most likely scoreline of a World Cup match — built with classical ML, XGBoost, a Dixon-Coles Poisson model, and PyTorch neural networks.

![Prediction Dashboard](ned_tun_wc2026_prediction.png)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Match Context](#-match-context)
- [Repository Structure](#-repository-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Feature Engineering](#-feature-engineering)
- [Models & Techniques](#-models--techniques)
  - [Machine Learning](#machine-learning-models)
  - [Deep Learning](#deep-learning-models-pytorch)
  - [Poisson Scoring Model](#dixon-coles-poisson-model)
  - [Ensemble Strategy](#ensemble-strategy)
- [Results](#-results)
- [How to Run](#-how-to-run)
- [How to Use Real Data](#-how-to-use-real-data)
- [Dependencies](#-dependencies)
- [Author](#-author)

---

## 🧠 Project Overview

This project demonstrates a complete ML/DL sports prediction system built from scratch. It covers:

- **Feature engineering** from raw match history using exponential decay weighting, head-to-head stats, and tournament context
- **4 classical ML models** (Logistic Regression, Random Forest, XGBoost, Gradient Boosting) with cross-validation and probability calibration
- **3 PyTorch deep learning models** (Feedforward Network, Dual-Head ScoreNet, FormLSTM)
- **Dixon-Coles Poisson model** for scoreline probability distributions
- **Weighted ensemble** combining all models into a final prediction
- **9-panel visualization dashboard** covering outcome probabilities, scoreline heatmap, feature importance, team radar chart, and more

The pipeline is fully modular — swap in real Kaggle data with a single line change and it works for any international match.

---

## 🏟️ Match Context

| | 🇳🇱 Netherlands | 🇹🇳 Tunisia |
|---|---|---|
| **FIFA Ranking** | #6 | #32 |
| **WC 2026 Points** | 4 pts | 0 pts ❌ Eliminated |
| **WC 2026 GD** | +4 | -8 |
| **MD1 Result** | 2-2 vs Japan | 1-5 vs Sweden |
| **MD2 Result** | 5-1 vs Sweden | 0-4 vs Japan |
| **Status** | Qualified (Round of 32) | Eliminated, manager sacked |

**Head-to-Head:** Netherlands won 1-0 vs Tunisia at the 2010 World Cup. Most recent meeting: 2-2 friendly draw (2013).

---

## 📁 Repository Structure

```
wc2026-match-predictor/
│
├── ned_tun_predictor.py          # Main ML pipeline (classical models + visualization)
├── deep_learning_pytorch.py      # Standalone PyTorch DL pipeline (3 neural networks)
├── notebook.ipynb                # Jupyter Notebook version of the full pipeline
├── ned_tun_wc2026_prediction.png # Output dashboard visualization
└── README.md                     # This file
```

> **Note:** The pipeline ships with a synthetic-but-realistic dataset so it runs out of the box. See [How to Use Real Data](#-how-to-use-real-data) to plug in the Kaggle dataset.

---

## 🏗️ Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    RAW MATCH HISTORY                          │
│         (2010–2026 international results)                    │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                          │
│  • Exponential decay form (last 10 matches)                  │
│  • Head-to-head historical stats                             │
│  • WC 2026 tournament context (pts, GD, elimination)         │
│  • FIFA ranking differential                                 │
│  → 40-dimensional feature vector per match                   │
└──────┬───────────────┬──────────────────┬────────────────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────┐ ┌──────────────┐ ┌────────────────────┐
│  Classical  │ │  PyTorch DL  │ │  Dixon-Coles       │
│  ML Models  │ │  Models      │ │  Poisson Model     │
│             │ │              │ │                    │
│ • LogReg    │ │ • FFN Net    │ │ • λ_home = f(att,  │
│ • Rand.For. │ │ • ScoreNet   │ │   def, form)       │
│ • XGBoost   │ │ • FormLSTM   │ │ • P(score=k|λ)     │
│ • Grad.Bst. │ │              │ │ • Scoreline grid   │
└──────┬──────┘ └──────┬───────┘ └─────────┬──────────┘
       │               │                   │
       └───────────────┴───────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  WEIGHTED ENSEMBLE    │
              │  (inverse log-loss)   │
              └───────────┬───────────┘
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
  P(Win / Draw / Loss)        Most Likely Scoreline
   + confidence bands          + Poisson probability grid
```

---

## 🔧 Feature Engineering

Each match is represented as a **40-dimensional feature vector** computed entirely from pre-match data (no leakage).

### Team Form Features (per team, exponential decay)
Computed over the last 10 matches before the match date, with decay factor `α = 0.9` so recent matches count more:

| Feature | Description |
|---|---|
| `form_pts` | Weighted avg points (W=3, D=1, L=0) |
| `goals_scored_avg` | Weighted avg goals scored |
| `goals_conceded_avg` | Weighted avg goals conceded |
| `win_rate` | Weighted win percentage |
| `draw_rate` | Weighted draw percentage |
| `clean_sheet_rate` | Weighted clean sheet percentage |
| `xG_for` / `xG_against` | Proxy expected goals (scaled from actuals) |
| `gd_avg` | Weighted average goal difference |

**Exponential Decay Formula:**
$$w_i = \alpha^{N-1-i}, \quad \hat{w}_i = \frac{w_i}{\sum_j w_j}$$

Where $i=0$ is the oldest match and $i=N-1$ is the most recent.

### Differential Features (most predictive)
The difference between home and away team values:
`form_pts_diff`, `goals_scored_diff`, `goals_conceded_diff`, `win_rate_diff`, `xG_diff`, `gd_diff`

### Head-to-Head Features
From all historical matchups between the two teams:
`h2h_win_rate`, `h2h_draw_rate`, `h2h_goals_home`, `h2h_goals_away`

### Rankings
`home_ranking`, `away_ranking`, `ranking_diff` (positive = home team favored)

### Tournament Context (key for WC predictions)
| Feature | NED | TUN |
|---|---|---|
| `wc_pts` | 4 | 0 |
| `wc_gd` | +4 | -8 |
| `wc_pts_diff` | +4 | — |
| `wc_gd_diff` | +12 | — |
| `eliminated` | 0 | 1 |
| `elimination_pressure` | +1 (TUN under pressure) | — |

---

## 🤖 Models & Techniques

### Machine Learning Models

All 4 models are trained on the feature matrix, evaluated with **5-fold stratified cross-validation**, and probabilities are **calibrated** using isotonic regression to ensure `P(Win) + P(Draw) + P(Loss) = 1` with well-calibrated confidence.

#### 1. Logistic Regression (Baseline)
```python
LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="lbfgs")
```
- Multinomial softmax over 3 classes
- L2 regularization (`C=0.5`) to prevent overfitting on small dataset
- `class_weight="balanced"` to handle draw underrepresentation
- Wrapped in `CalibratedClassifierCV` with isotonic regression

#### 2. Random Forest
```python
RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                       min_samples_leaf=3, random_state=42)
```
- 300 decision trees with max depth 6 to prevent overfitting
- Bagging + feature subsampling for variance reduction
- Returns averaged probability distributions across all trees

#### 3. XGBoost
```python
xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                  n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8)
```
- Gradient-boosted trees with softmax output (`multi:softprob`)
- Low learning rate + high n_estimators for better generalization
- Column and row subsampling to reduce overfitting
- Also used for **feature importance** ranking

#### 4. Gradient Boosting (sklearn)
```python
GradientBoostingClassifier(n_estimators=200, max_depth=3,
                            learning_rate=0.05, subsample=0.8)
```
- Slower but often more stable than XGBoost on small datasets
- Shallow trees (depth=3) as weak learners

**Evaluation Metric:** Log-Loss (cross-entropy) — measures quality of probability distributions, not just accuracy.

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \{W,D,L\}} y_{ic} \log p_{ic}$$

---

### Deep Learning Models (PyTorch)

> File: `deep_learning_pytorch.py` — runs completely standalone.

#### Model A: MatchOutcomeNet (Feedforward Neural Network)
```
Input (40) → Linear(128) → BatchNorm → ReLU → Dropout(0.35)
           → Linear(64)  → BatchNorm → ReLU → Dropout(0.25)
           → Linear(32)  → ReLU
           → Linear(3)   → [Win, Draw, Loss logits]
```
- **BatchNorm** after each layer for training stability
- **Dropout** (0.35, 0.25) to prevent overfitting on small football datasets
- **CrossEntropyLoss** with class weights `[1.0, 1.5, 1.2]` (draws upweighted)
- **CosineAnnealingLR** scheduler for smooth learning rate decay
- **Gradient clipping** (`max_norm=1.0`) to prevent exploding gradients

#### Model B: ScoreNet (Dual-Head Poisson Network)
```
Input (40) → Shared Backbone (Linear 64 → ReLU → Dropout → Linear 32 → ReLU)
           ├── Head 1: Linear(1) → Softplus() → λ_home (NED expected goals)
           └── Head 2: Linear(1) → Softplus() → λ_away (TUN expected goals)
```
- **Softplus activation** (`log(1 + e^x)`) enforces `λ > 0` (goals can't be negative)
- **Poisson NLL Loss**: $\mathcal{L} = \lambda - k \cdot \log(\lambda)$ per team
- Output `λ` feeds directly into the Dixon-Coles scoreline grid

#### Model C: FormLSTM (Sequence Model)
```
Input: (batch, seq_len=10, 6 features per match)
  Features: [goals_for, goals_against, is_win, is_draw, goal_diff, opp_rank_norm]

→ LSTM(hidden=32, num_layers=2, dropout=0.2, batch_first=True)
→ Take last hidden state hn[-1]
→ Linear(32 → 3) → [Win, Draw, Loss logits]
```
- Treats last 10 matches as a **time series** — captures momentum and form streaks
- 2-layer stacked LSTM for richer sequence representations
- Shorter sequences are **zero-padded** on the left
- Trained independently per team (home team's sequence used for prediction)

**Training config (all DL models):**

| Setting | Value |
|---|---|
| Optimizer | Adam (`lr=1e-3`, `weight_decay=1e-4`) |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Epochs | FFN: 150 · ScoreNet: 120 · LSTM: 100 |
| Device | Auto-detects CUDA/CPU |

---

### Dixon-Coles Poisson Model

The industry-standard approach for football scoreline prediction. Models home and away goals as **independent Poisson processes**:

$$P(\text{NED scores } i, \text{ TUN scores } j) = \text{Poisson}(i;\lambda_{NED}) \times \text{Poisson}(j;\lambda_{TUN})$$

Where the expected goals (λ) are estimated from:

$$\lambda_{home} = \frac{\mu_{att,home}}{\bar{g}} \times \frac{\mu_{def,away}}{\bar{g}} \times \bar{g} \times \text{form\_multiplier}$$

- $\bar{g}$ = league average goals per team per match (1.35 for WC)
- `form_multiplier` applies a WC 2026 tournament form boost/penalty:
  - Netherlands: ×1.15 (dominant in tournament)
  - Tunisia: ×0.72 (eliminated, heavy losses, manager sacked)
- A full `7×7` scoreline probability grid is computed, then outcome probabilities are derived by summing the appropriate cells

---

### Ensemble Strategy

The final prediction blends all models using a **weighted average**, where weights are assigned by inverse cross-validation log-loss (better models get higher weight):

$$w_m = \frac{1/\mathcal{L}_m}{\sum_k 1/\mathcal{L}_k}$$

**Final blend:**
$$P_{final} = 0.60 \times P_{ML\text{-}ensemble} + 0.40 \times P_{Poisson}$$

This gives the classical ML models stronger influence on outcome probabilities while the Poisson model anchors the scoreline predictions.

---

## 📊 Results

### Outcome Probabilities

| Model | NED Win | Draw | TUN Win |
|---|---|---|---|
| Logistic Regression | 75.6% | 11.3% | 13.1% |
| Random Forest | 60.9% | 21.7% | 17.4% |
| XGBoost | 81.8% | 5.6% | 12.7% |
| Gradient Boosting | 46.9% | 11.7% | 41.4% |
| MatchOutcomeNet (FFN) | 85.3% | 3.9% | 10.8% |
| ScoreNet / Poisson | 80.5% | 13.5% | 4.8% |
| FormLSTM | 7.8% | 90.8% | 1.4% |
| **Final Ensemble** | **~72–77%** | **~10–29%** | **~6–13%** |

> The FormLSTM's high draw prediction is expected — it only sees NED's form sequence (recent draws against Japan and form variability), not Tunisia's context. The FFN and Poisson models, which see all 40 features including WC context, are the stronger signals.

### Most Likely Scorelines (Poisson / ScoreNet)

| Rank | Score | Probability | Result |
|---|---|---|---|
| ⭐ 1 | **NED 2–0 TUN** | 17.2% | NED Win |
| 2 | NED 1–0 TUN | 14.3% | NED Win |
| 3 | NED 3–0 TUN | 13.8% | NED Win |
| 4 | NED 4–0 TUN | 8.3% | NED Win |
| 5 | NED 2–1 TUN | 7.2% | NED Win |

**Expected Goals:** NED 2.41 · TUN 0.42

---

## 🚀 How to Run

### Option 1: Classical ML Pipeline (`ned_tun_predictor.py`)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/wc2026-match-predictor.git
cd wc2026-match-predictor

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy

# 3. Run the pipeline
python ned_tun_predictor.py
```

**Output:** Prints predictions to console + saves `ned_tun_wc2026_prediction.png` dashboard.

---

### Option 2: Deep Learning Pipeline (`deep_learning_pytorch.py`)

```bash
# 1. Install PyTorch (CPU)
pip install torch pandas numpy scikit-learn scipy

# 2. Run (fully standalone — no other files needed)
python deep_learning_pytorch.py
```

**Output:** Trains 3 neural networks, prints predictions, saves `.pt` model weights.

---

### Option 3: Jupyter Notebook (`notebook.ipynb`)

```bash
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn scipy torch
jupyter notebook notebook.ipynb
```
Then run all cells (`Kernel → Restart & Run All`).

---

### Option 4: Google Colab (free GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `deep_learning_pytorch.py` or `notebook.ipynb`
3. Set runtime: `Runtime → Change runtime type → T4 GPU`
4. Run all cells

PyTorch will automatically detect and use the GPU.

---

## 📦 How to Use Real Data

The pipeline ships with a synthetic dataset for reproducibility. To use the real Kaggle dataset:

**Step 1:** Download from Kaggle:
- [International Football Results 1872–2023](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [FIFA World Rankings](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)

**Step 2:** In `ned_tun_predictor.py`, replace the `build_historical_dataset()` call with:

```python
# Load match results
df_raw = pd.read_csv('results.csv')
df_raw['date'] = pd.to_datetime(df_raw['date']).dt.date
df_raw = df_raw[df_raw['date'] >= date(2010, 1, 1)].reset_index(drop=True)

# Add FIFA rankings (merge on team name + date)
rankings = pd.read_csv('fifa_ranking.csv')
# ... merge logic to add home_ranking / away_ranking columns

# Manually append WC 2026 Group F results (not in Kaggle yet)
wc_2026 = pd.DataFrame([...])  # see script for format
df_raw = pd.concat([df_raw, wc_2026]).sort_values('date').reset_index(drop=True)
```

**Step 3:** Add real xG data (optional but recommended):
- Source: [FBref](https://fbref.com), [Understat](https://understat.com), or [StatsBomb Open Data](https://github.com/statsbomb/open-data)

---

## 📐 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥1.5 | Data manipulation |
| `numpy` | ≥1.23 | Numerical computing |
| `scikit-learn` | ≥1.2 | ML models, cross-validation, calibration |
| `xgboost` | ≥1.7 | Gradient boosting classifier |
| `scipy` | ≥1.9 | Poisson distribution, softmax |
| `matplotlib` | ≥3.6 | Visualization |
| `seaborn` | ≥0.12 | Heatmap styling |
| `torch` (PyTorch) | ≥2.0 | Neural network training (DL file only) |

Install all at once:
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn torch
```

---

## 👤 Author

**Ghaith Hajji**
3rd-year IT & Business Analytics student · Tunis Business School
Specialization: Data Science & Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/ghaith-hajji-b40572312/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/VECTORZ24)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

> **Disclaimer:** This project is built for educational and portfolio purposes. Predictions are probabilistic estimates based on historical data and should not be used for betting or financial decisions.
