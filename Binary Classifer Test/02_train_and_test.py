"""Step 2 (row-by-row voting): the model names every single reading, then
each group of WINDOW consecutive readings gets the most common answer (a vote).

Run:  python 02_train_and_test.py   ->  charts/2_results_table_voting.png
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, MODES, OUT, RIGHT_AT, SEED, WINDOW, draw_table, load, test_mask


def make_models():
    return {
        "Logistic\nregression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random\nforest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
        "Neural\nnetwork": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                        max_iter=300, random_state=SEED)),
    }


if __name__ == "__main__":
    df = load()
    X, y = df[list(FEATURES)], df["Fault"]
    test = test_mask(y)
    X_tr, X_te, y_tr, y_te = X[~test], X[test], y[~test], y[test]
    print(f"train rows: {len(X_tr):,}   test rows: {len(X_te):,}")

    results = {}
    for name, model in make_models().items():
        pred = pd.Series(model.fit(X_tr, y_tr).predict(X_te), index=X_te.index)
        results[name] = {}
        for m in MODES:
            p = pred[y_te == m].to_numpy()
            n_win = len(p) // WINDOW
            votes = np.array([np.bincount(p[k * WINDOW:(k + 1) * WINDOW], minlength=4).argmax() for k in range(n_win)])
            wrong = votes[votes != m]
            confused = int(np.bincount(wrong, minlength=4).argmax()) if len(wrong) else None
            results[name][m] = (int((votes == m).sum()), n_win, confused)
        print(name.replace("\n", " "), {m: f"{a}/{b}" for m, (a, b, _) in results[name].items()})

    draw_table(
        results,
        "Approach A: name every reading, then vote",
        f"Each test = {WINDOW} consecutive readings from one failure mode. The model names each reading separately,\n"
        f"and the most common answer is its answer. RIGHT = correct in at least {RIGHT_AT:.0%} of that mode's tests.",
        f"Tests come from the last 20% of each failure mode ({len(X_te):,} readings the models never saw in training).",
        OUT / "2_results_table_voting.png")
