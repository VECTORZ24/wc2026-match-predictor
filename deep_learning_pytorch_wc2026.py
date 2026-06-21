"""
==============================================================================
 Deep Learning: PyTorch Match Predictor — Netherlands vs Tunisia (WC 2026)
 FULLY STANDALONE — runs on its own, no external dependencies needed.
 
 Install requirements:
   pip install torch pandas numpy scikit-learn xgboost scipy
 
 Or run for free on Google Colab:
   https://colab.research.google.com  (File → Upload → this file → Run All)
==============================================================================
"""

# ─────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import warnings
from datetime import date
from scipy.stats import poisson

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

print("✅ Imports OK")
print(f"   PyTorch version : {torch.__version__}")
print(f"   Device          : {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}\n")

# ─────────────────────────────────────────────────────────
# 1. BUILD DATASET  (same logic as ned_tun_predictor.py)
#    → swap in real Kaggle CSV here if you have it
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Building Historical Dataset")
print("=" * 60)

def build_dataset():
    """
    Builds a synthetic-but-realistic dataset of ~300 international
    matches (2010-2026) for NED, TUN, and 8 peer nations.

    TO USE REAL DATA instead:
        df = pd.read_csv('results.csv')   # Kaggle: International Football Results
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df[df['date'] >= date(2010, 1, 1)].reset_index(drop=True)
        # Add home_ranking / away_ranking columns from FIFA ranking CSV
    """
    np.random.seed(42)
    teams = {
        "Netherlands": {"attack": 1.85, "defense": 0.75, "ranking": 6},
        "Tunisia":     {"attack": 0.95, "defense": 1.10, "ranking": 32},
        "Japan":       {"attack": 1.30, "defense": 0.85, "ranking": 18},
        "Sweden":      {"attack": 1.50, "defense": 0.90, "ranking": 20},
        "Germany":     {"attack": 1.90, "defense": 0.70, "ranking": 4},
        "France":      {"attack": 2.00, "defense": 0.65, "ranking": 2},
        "Brazil":      {"attack": 1.80, "defense": 0.80, "ranking": 5},
        "Senegal":     {"attack": 1.10, "defense": 1.05, "ranking": 20},
        "Morocco":     {"attack": 1.20, "defense": 0.95, "ranking": 14},
        "Spain":       {"attack": 1.75, "defense": 0.72, "ranking": 8},
    }
    team_list = list(teams.keys())
    rows = []
    base = date(2010, 1, 1)
    n = 300
    for i in range(n):
        home, away = np.random.choice(team_list, 2, replace=False)
        h, a = teams[home], teams[away]
        lh = max(0.3, h["attack"] * a["defense"] * np.random.uniform(0.7, 1.3))
        la = max(0.3, a["attack"] * h["defense"] * np.random.uniform(0.7, 1.3))
        gh, ga = np.random.poisson(lh), np.random.poisson(la)
        outcome = "home_win" if gh > ga else ("away_win" if ga > gh else "draw")
        d = base + pd.Timedelta(days=int(i * (16 * 365 / n)))
        rows.append({"date": d, "home_team": home, "away_team": away,
                     "home_score": gh, "away_score": ga, "outcome": outcome,
                     "home_ranking": h["ranking"], "away_ranking": a["ranking"]})

    # Real WC 2026 Group F results (injected)
    for r in [
        {"date": date(2026,6,14), "home_team":"Japan",       "away_team":"Netherlands",
         "home_score":2, "away_score":2, "outcome":"draw",
         "home_ranking":18, "away_ranking":6},
        {"date": date(2026,6,14), "home_team":"Sweden",      "away_team":"Tunisia",
         "home_score":5, "away_score":1, "outcome":"home_win",
         "home_ranking":20, "away_ranking":32},
        {"date": date(2026,6,20), "home_team":"Netherlands", "away_team":"Sweden",
         "home_score":5, "away_score":1, "outcome":"home_win",
         "home_ranking":6, "away_ranking":20},
        {"date": date(2026,6,21), "home_team":"Japan",       "away_team":"Tunisia",
         "home_score":4, "away_score":0, "outcome":"home_win",
         "home_ranking":18, "away_ranking":32},
    ]:
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"  ✓ {len(df)} matches  ({df['date'].min()} → {df['date'].max()})")
    return df

df_raw = build_dataset()

# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Feature Engineering")
print("=" * 60)

LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

def team_form(df, team, before_date, n=10, decay=0.9):
    mask = ((df["home_team"]==team)|(df["away_team"]==team)) & (df["date"]<before_date)
    m = df[mask].sort_values("date").tail(n)
    if len(m) == 0:
        return dict(fp=1.2, gs=1.2, gc=1.2, wr=0.4, dr=0.3, cs=0.25, gd=0.0)
    w = np.array([decay**(len(m)-1-i) for i in range(len(m))]); w /= w.sum()
    fp, gs, gc, wr, dr, cs = [], [], [], [], [], []
    for _, r in m.iterrows():
        ih = r["home_team"] == team
        gf = r["home_score"] if ih else r["away_score"]
        ga = r["away_score"] if ih else r["home_score"]
        win = gf > ga; draw = gf == ga
        fp.append(3 if win else (1 if draw else 0))
        gs.append(gf); gc.append(ga)
        wr.append(int(win)); dr.append(int(draw)); cs.append(int(ga==0))
    fp,gs,gc,wr,dr,cs = map(np.array,[fp,gs,gc,wr,dr,cs])
    return dict(fp=np.dot(w,fp), gs=np.dot(w,gs), gc=np.dot(w,gc),
                wr=np.dot(w,wr), dr=np.dot(w,dr), cs=np.dot(w,cs),
                gd=np.dot(w,gs-gc))

def h2h_stats(df, t1, t2, before_date, n=10):
    mask = (((df["home_team"]==t1)&(df["away_team"]==t2))|
            ((df["home_team"]==t2)&(df["away_team"]==t1))) & (df["date"]<before_date)
    m = df[mask].tail(n)
    if len(m) == 0:
        return dict(wr=0.5, dr=0.2, g1=1.2, g2=1.2)
    wins, draws, g1, g2 = 0, 0, [], []
    for _, r in m.iterrows():
        ih = r["home_team"] == t1
        gf = r["home_score"] if ih else r["away_score"]
        ga = r["away_score"] if ih else r["home_score"]
        if gf > ga: wins += 1
        elif gf == ga: draws += 1
        g1.append(gf); g2.append(ga)
    return dict(wr=wins/len(m), dr=draws/len(m), g1=np.mean(g1), g2=np.mean(g2))

def make_features(df, ht, at, d, hr, ar,
                  hwp=0, awp=0, hgd=0, agd=0, helim=0, aelim=0):
    h = team_form(df, ht, d); a = team_form(df, at, d)
    hh = h2h_stats(df, ht, at, d)
    return {
        "h_fp": h["fp"],  "h_gs": h["gs"],  "h_gc": h["gc"],
        "h_wr": h["wr"],  "h_dr": h["dr"],  "h_cs": h["cs"],  "h_gd": h["gd"],
        "a_fp": a["fp"],  "a_gs": a["gs"],  "a_gc": a["gc"],
        "a_wr": a["wr"],  "a_dr": a["dr"],  "a_cs": a["cs"],  "a_gd": a["gd"],
        "diff_fp": h["fp"]-a["fp"],  "diff_gs": h["gs"]-a["gs"],
        "diff_gc": h["gc"]-a["gc"],  "diff_wr": h["wr"]-a["wr"],
        "diff_gd": h["gd"]-a["gd"],
        "h_rank": hr,  "a_rank": ar,  "rank_diff": ar-hr,
        "h2h_wr": hh["wr"],  "h2h_dr": hh["dr"],
        "h2h_g1": hh["g1"], "h2h_g2": hh["g2"],
        "h_wcp": hwp, "a_wcp": awp, "diff_wcp": hwp-awp,
        "h_wcgd": hgd, "a_wcgd": agd, "diff_wcgd": hgd-agd,
        "h_elim": helim, "a_elim": aelim, "elim_pressure": aelim-helim,
    }

# Build training matrix
rows_X, rows_y = [], []
for i, row in df_raw.iterrows():
    if i < 20: continue
    rows_X.append(make_features(df_raw, row["home_team"], row["away_team"],
                                row["date"], row["home_ranking"], row["away_ranking"]))
    rows_y.append(LABEL_MAP[row["outcome"]])

X_df = pd.DataFrame(rows_X)
y    = np.array(rows_y)
FEATURE_NAMES = list(X_df.columns)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_df).astype(np.float32)
print(f"  ✓ Feature matrix : {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")
print(f"  ✓ Labels         : Win={np.sum(y==0)}  Draw={np.sum(y==1)}  Loss={np.sum(y==2)}")

# Target match: Netherlands vs Tunisia, Jun 25 2026
target_feat = make_features(
    df_raw,
    ht="Netherlands", at="Tunisia",
    d=date(2026,6,25), hr=6, ar=32,
    hwp=4, awp=0, hgd=4, agd=-8,
    helim=0, aelim=1
)
X_target = scaler.transform(pd.DataFrame([target_feat])).astype(np.float32)

# ─────────────────────────────────────────────────────────
# 3. PYTORCH MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — PyTorch Model Definitions")
print("=" * 60)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = X_scaled.shape[1]

# ── Model A: Feedforward Outcome Classifier ──
class MatchOutcomeNet(nn.Module):
    """
    Input  : feature vector (40-dim)
    Output : logits for [Win, Draw, Loss]  → apply softmax for probabilities
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),         # 3 classes: Win / Draw / Loss
        )
    def forward(self, x):
        return self.net(x)            # raw logits (use CrossEntropyLoss in training)

# ── Model B: Dual-Head Score Predictor ──
class ScoreNet(nn.Module):
    """
    Input  : same feature vector
    Output : (lambda_home, lambda_away) — Poisson rate for each team's goals
    Use poisson.pmf(k, lambda) to get P(team scores exactly k goals)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),        nn.ReLU(),
        )
        self.head_home = nn.Sequential(nn.Linear(32, 1), nn.Softplus())  # λ ≥ 0
        self.head_away = nn.Sequential(nn.Linear(32, 1), nn.Softplus())  # λ ≥ 0

    def forward(self, x):
        s = self.shared(x)
        return self.head_home(s).squeeze(1), self.head_away(s).squeeze(1)

# ── Model C: LSTM over last-N-match form sequence ──
class FormLSTM(nn.Module):
    """
    Input  : (batch, seq_len=10, features=6) — one row per recent match
             features = [goals_for, goals_against, is_win, is_draw, gd, opp_ranking_norm]
    Output : logits for [Win, Draw, Loss]
    """
    def __init__(self, input_size=6, hidden=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 3)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)   # hn: (num_layers, batch, hidden)
        return self.fc(hn[-1])       # last layer's hidden state

print(f"  ✓ MatchOutcomeNet  — 128→64→32→3  (classifier)")
print(f"  ✓ ScoreNet         — shared 64→32, dual heads (Poisson λ)")
print(f"  ✓ FormLSTM         — 2-layer LSTM → 3 classes")
print(f"  ✓ All models will run on: {DEVICE}")

# ─────────────────────────────────────────────────────────
# 4. TRAINING — MatchOutcomeNet
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Training MatchOutcomeNet")
print("=" * 60)

def train_outcome_net(X_train, y_train, epochs=150, lr=1e-3, batch_size=32):
    model = MatchOutcomeNet(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class weights to handle draw imbalance
    cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    )

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  Loss: {avg_loss:.4f}")

    return model, history


outcome_model, loss_history = train_outcome_net(X_scaled, y, epochs=150)

# ─────────────────────────────────────────────────────────
# 5. TRAINING — ScoreNet (Poisson loss)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Training ScoreNet (Poisson Loss)")
print("=" * 60)

def poisson_nll_loss(lam, k):
    """Negative log-likelihood of Poisson: -[k*log(lam) - lam - log(k!)]"""
    return (lam - k * torch.log(lam + 1e-8)).mean()

def train_score_net(X_train, df_train, epochs=120, lr=1e-3):
    model = ScoreNet(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Build score targets aligned to training rows (skip first 20 rows)
    score_rows = df_train.iloc[20:].reset_index(drop=True)
    y_home = torch.tensor(score_rows["home_score"].values, dtype=torch.float32).to(DEVICE)
    y_away = torch.tensor(score_rows["away_score"].values, dtype=torch.float32).to(DEVICE)
    X_t    = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        lam_h, lam_a = model(X_t)
        loss = poisson_nll_loss(lam_h, y_home) + poisson_nll_loss(lam_a, y_away)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  Poisson NLL: {loss.item():.4f}")

    return model, history

score_model, score_loss_history = train_score_net(X_scaled, df_raw)

# ─────────────────────────────────────────────────────────
# 6. TRAINING — FormLSTM
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Training FormLSTM (sequence model)")
print("=" * 60)

def build_lstm_sequences(df, team_list, before_date, seq_len=10):
    """
    For each team, build a (seq_len, 6) matrix of last N match features.
    features per match: [goals_for, goals_against, is_win, is_draw, gd, opp_rank_norm]
    """
    sequences, labels = [], []
    for _, row in df[df["date"] < before_date].iterrows():
        for team, label_key in [(row["home_team"], "home"), (row["away_team"], "away")]:
            mask = ((df["home_team"]==team)|(df["away_team"]==team)) & (df["date"]<row["date"])
            history = df[mask].sort_values("date").tail(seq_len)
            if len(history) < 3: continue

            seq = []
            for _, h in history.iterrows():
                ih = h["home_team"] == team
                gf = h["home_score"] if ih else h["away_score"]
                ga = h["away_score"] if ih else h["home_score"]
                opp_rank = h["away_ranking"] if ih else h["home_ranking"]
                seq.append([gf, ga, int(gf>ga), int(gf==ga), gf-ga, opp_rank/50.0])

            # Pad if shorter than seq_len
            while len(seq) < seq_len:
                seq.insert(0, [0.0]*6)

            outcome = row["outcome"]
            if label_key == "away":
                outcome = "away_win" if outcome=="home_win" else ("home_win" if outcome=="away_win" else "draw")
            lbl = LABEL_MAP.get(outcome if label_key=="home" else
                                 ("home_win" if outcome=="away_win" else
                                  ("away_win" if outcome=="home_win" else "draw")))
            if lbl is not None:
                sequences.append(seq[-seq_len:])
                labels.append(lbl)

    return np.array(sequences, dtype=np.float32), np.array(labels)

print("  Building LSTM sequences...")
team_list = df_raw["home_team"].unique().tolist()
X_seq, y_seq = build_lstm_sequences(df_raw, team_list, date(2026, 6, 25))
print(f"  ✓ Sequences: {X_seq.shape}  Labels: {y_seq.shape}")

def train_lstm(X_seq, y_seq, epochs=100, lr=1e-3):
    model = FormLSTM(input_size=6, hidden=32, num_layers=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_seq)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))

    X_t = torch.tensor(X_seq).to(DEVICE)
    y_t = torch.tensor(y_seq, dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / len(loader))
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  Loss: {total_loss/len(loader):.4f}")

    return model, history

lstm_model, lstm_loss_history = train_lstm(X_seq, y_seq, epochs=100)

# ─────────────────────────────────────────────────────────
# 7. PREDICT — Netherlands vs Tunisia
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Predicting: Netherlands vs Tunisia (Jun 25, 2026)")
print("=" * 60)

def predict_outcome(model, X):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(x_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

def predict_score(model, X):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        lam_h, lam_a = model(x_t)
    return lam_h.cpu().numpy()[0], lam_a.cpu().numpy()[0]

# ── MatchOutcomeNet prediction ──
ffn_probs = predict_outcome(outcome_model, X_target)[0]
print(f"\n  MatchOutcomeNet (FFN)")
print(f"    NED Win : {ffn_probs[0]:.1%}")
print(f"    Draw    : {ffn_probs[1]:.1%}")
print(f"    TUN Win : {ffn_probs[2]:.1%}")

# ── ScoreNet prediction ──
lam_ned, lam_tun = predict_score(score_model, X_target)
print(f"\n  ScoreNet (Poisson λ)")
print(f"    λ NED (expected goals) : {lam_ned:.2f}")
print(f"    λ TUN (expected goals) : {lam_tun:.2f}")

# Build scoreline grid from Poisson lambdas
MAX_G = 7
score_grid = np.zeros((MAX_G, MAX_G))
for i in range(MAX_G):
    for j in range(MAX_G):
        score_grid[i,j] = poisson.pmf(i, max(0.3, float(lam_ned))) * \
                          poisson.pmf(j, max(0.1, float(lam_tun)))

p_ned_win = sum(score_grid[i,j] for i in range(MAX_G) for j in range(MAX_G) if i>j)
p_draw    = sum(score_grid[i,j] for i in range(MAX_G) for j in range(MAX_G) if i==j)
p_tun_win = sum(score_grid[i,j] for i in range(MAX_G) for j in range(MAX_G) if i<j)

flat = sorted([(score_grid[i,j],i,j) for i in range(MAX_G) for j in range(MAX_G)], reverse=True)
print(f"\n  Poisson outcome probs: NED Win={p_ned_win:.1%}  Draw={p_draw:.1%}  TUN Win={p_tun_win:.1%}")
print(f"\n  Top 5 most likely scorelines:")
for prob, g_ned, g_tun in flat[:5]:
    tag = "NED Win" if g_ned>g_tun else ("Draw" if g_ned==g_tun else "TUN Win")
    print(f"    NED {g_ned}-{g_tun} TUN  →  {prob:.2%}  ({tag})")

# ── FormLSTM prediction (build NED's recent sequence) ──
def build_team_sequence(df, team, before_date, seq_len=10):
    mask = ((df["home_team"]==team)|(df["away_team"]==team)) & (df["date"]<before_date)
    history = df[mask].sort_values("date").tail(seq_len)
    seq = []
    for _, h in history.iterrows():
        ih = h["home_team"] == team
        gf = h["home_score"] if ih else h["away_score"]
        ga = h["away_score"] if ih else h["home_score"]
        opp_rank = h["away_ranking"] if ih else h["home_ranking"]
        seq.append([gf, ga, int(gf>ga), int(gf==ga), gf-ga, opp_rank/50.0])
    while len(seq) < seq_len:
        seq.insert(0, [0.0]*6)
    return np.array([seq[-seq_len:]], dtype=np.float32)

ned_seq = build_team_sequence(df_raw, "Netherlands", date(2026,6,25))
lstm_probs = predict_outcome(lstm_model, ned_seq)[0]
print(f"\n  FormLSTM (NED form sequence)")
print(f"    NED Win : {lstm_probs[0]:.1%}")
print(f"    Draw    : {lstm_probs[1]:.1%}")
print(f"    TUN Win : {lstm_probs[2]:.1%}")

# ─────────────────────────────────────────────────────────
# 8. ENSEMBLE ALL THREE DL MODELS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Deep Learning Ensemble")
print("=" * 60)

poisson_probs = np.array([p_ned_win, p_draw, p_tun_win])
dl_ensemble = (
    0.40 * ffn_probs +
    0.35 * poisson_probs +
    0.25 * lstm_probs
)
dl_ensemble /= dl_ensemble.sum()

print(f"""
  ┌══════════════════════════════════════════════════════════┐
  │   DEEP LEARNING ENSEMBLE — NED vs TUN (Jun 25, 2026)    │
  ╠══════════════════════════════════════════════════════════╣
  │                                                          │
  │   Model            NED Win   Draw   TUN Win             │
  │   ─────────────────────────────────────────             │
  │   FFN (Outcome)    {ffn_probs[0]:>6.1%}  {ffn_probs[1]:>6.1%}  {ffn_probs[2]:>6.1%}          │
  │   Poisson (Score)  {poisson_probs[0]:>6.1%}  {poisson_probs[1]:>6.1%}  {poisson_probs[2]:>6.1%}          │
  │   FormLSTM         {lstm_probs[0]:>6.1%}  {lstm_probs[1]:>6.1%}  {lstm_probs[2]:>6.1%}          │
  │   ─────────────────────────────────────────             │
  │   DL ENSEMBLE      {dl_ensemble[0]:>6.1%}  {dl_ensemble[1]:>6.1%}  {dl_ensemble[2]:>6.1%}          │
  │                                                          │
  │   ⭐ Most likely: NED {flat[0][1]}-{flat[0][2]} TUN ({flat[0][0]:.1%})             │
  │   ⭐ Runner-up  : NED {flat[1][1]}-{flat[1][2]} TUN ({flat[1][0]:.1%})             │
  ╠══════════════════════════════════════════════════════════╣
  │   CONTEXT: Tunisia ELIMINATED (0pts, -8 GD)             │
  │            Netherlands Qualified (4pts, +4 GD)          │
  └══════════════════════════════════════════════════════════┘
""")

# ─────────────────────────────────────────────────────────
# 9. SAVE MODELS
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 9 — Saving Models")
print("=" * 60)

torch.save(outcome_model.state_dict(), "outcome_net.pt")
torch.save(score_model.state_dict(),   "score_net.pt")
torch.save(lstm_model.state_dict(),    "form_lstm.pt")
print("  ✓ outcome_net.pt   — MatchOutcomeNet weights")
print("  ✓ score_net.pt     — ScoreNet weights")
print("  ✓ form_lstm.pt     — FormLSTM weights")

print("""
  To reload and reuse later:
    model = MatchOutcomeNet(input_dim=40)
    model.load_state_dict(torch.load('outcome_net.pt'))
    model.eval()
""")

print("=" * 60)
print("✅ DEEP LEARNING PIPELINE COMPLETE!")
print("=" * 60)
