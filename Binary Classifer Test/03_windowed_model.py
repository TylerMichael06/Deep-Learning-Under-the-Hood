"""Step 3 (group first): summarize each group of WINDOW consecutive readings
into its average and spread (standard deviation) for every sensor, then train
the model on those summaries. One summary in -> one answer out.

Run:  python 03_windowed_model.py   ->  charts/3_results_table_windowed.png
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, MODES, OUT, RIGHT_AT, SEED, WINDOW, draw_table, load, test_mask

TRAIN_STRIDE = 10  # training windows overlap (start every 10 rows) to get more examples; test windows never overlap


def summarize(rows: pd.DataFrame) -> np.ndarray:
    """One window of readings -> [mean of each sensor..., std of each sensor...]."""
    return np.concatenate([rows.mean().to_numpy(), rows.std().to_numpy()])


def make_windows(X, y, mask, stride):
    """Cut each failure mode's rows (those where mask is True) into windows.
    Returns summaries, labels."""
    feats, labels = [], []
    for m in MODES:
        rows = X[(y == m).to_numpy() & mask]
        for start in range(0, len(rows) - WINDOW + 1, stride):
            feats.append(summarize(rows.iloc[start:start + WINDOW]))
            labels.append(m)
    return np.array(feats), np.array(labels)


def make_models():
    return {
        "Logistic\nregression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Random\nforest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "Neural\nnetwork": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
                                                                        max_iter=500, random_state=SEED)),
    }


if __name__ == "__main__":
    df = load()
    X, y = df[list(FEATURES)], df["Fault"]
    test = test_mask(y)
    F_tr, L_tr = make_windows(X, y, ~test, TRAIN_STRIDE)
    F_te, L_te = make_windows(X, y, test, WINDOW)
    print(f"train windows: {len(F_tr):,} (overlapping)   test windows: {len(F_te):,}")

    results = {}
    for name, model in make_models().items():
        pred = model.fit(F_tr, L_tr).predict(F_te)
        results[name] = {}
        for m in MODES:
            p = pred[L_te == m]
            wrong = p[p != m]
            confused = int(np.bincount(wrong, minlength=4).argmax()) if len(wrong) else None
            results[name][m] = (int((p == m).sum()), len(p), confused)
        print(name.replace("\n", " "), {m: f"{a}/{b}" for m, (a, b, _) in results[name].items()})

    draw_table(
        results,
        "Approach B: summarize first, then name the mode",
        f"Each test = {WINDOW} consecutive readings from one failure mode, boiled down to each sensor's average and spread.\n"
        f"The model sees only those summaries. RIGHT = correct in at least {RIGHT_AT:.0%} of that mode's tests.",
        f"Same held-out data as Approach A: the last 20% of each failure mode. {len(F_tr):,} overlapping windows used for training.",
        OUT / "3_results_table_windowed.png")
