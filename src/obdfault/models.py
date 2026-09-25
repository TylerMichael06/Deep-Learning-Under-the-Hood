import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn


def window_features(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def baseline(name, seed):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if name == "gbm":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")


class Standardize(nn.Module):
    def __init__(self, X_train):
        super().__init__()
        flat = torch.as_tensor(X_train.reshape(-1, X_train.shape[-1]), dtype=torch.float32)
        self.register_buffer("mean", flat.mean(0))
        self.register_buffer("std", flat.std(0).clamp_min(1e-6))

    def forward(self, x):
        return (x - self.mean) / self.std


class MLP(nn.Module):
    def __init__(self, X_train, hidden=(64, 32), dropout=0.1):
        super().__init__()
        layers, width = [Standardize(X_train)], X_train.shape[1]
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNN1d(nn.Module):
    def __init__(self, X_train, channels=32, dropout=0.1):
        super().__init__()
        self.norm = Standardize(X_train)
        self.conv = nn.Sequential(
            nn.Conv1d(X_train.shape[-1], channels, 7, padding=3), nn.ReLU(),
            nn.Conv1d(channels, channels, 7, padding=3, stride=2), nn.ReLU(),
            nn.Conv1d(channels, channels, 7, padding=3, stride=2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 1))

    def forward(self, x):
        return self.head(self.conv(self.norm(x).transpose(1, 2))).squeeze(-1)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batches(X, y, batch_size, device, order=None):
    order = np.arange(len(X)) if order is None else order
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield (torch.as_tensor(X[idx], dtype=torch.float32, device=device),
               torch.as_tensor(y[idx], dtype=torch.float32, device=device))


def train_torch(model, fold, *, epochs=30, batch_size=512, lr=1e-3, patience=5, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = _device()
    model.to(device)
    pos = int(fold.y_train.sum())
    pos_weight = torch.tensor((len(fold.y_train) - pos) / max(pos, 1), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, stale, history = float("inf"), None, 0, []
    for _ in range(epochs):
        model.train()
        for xb, yb in _batches(fold.X_train, fold.y_train, batch_size, device, rng.permutation(len(fold.X_train))):
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            total = sum(loss_fn(model(xb), yb).item() * len(xb)
                        for xb, yb in _batches(fold.X_val, fold.y_val, batch_size, device))
        history.append(total / len(fold.X_val))
        if history[-1] < best - 1e-4:
            best, stale = history[-1], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def predict_proba(model, X, batch_size=1024):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            torch.sigmoid(model(torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device))).cpu().numpy()
            for i in range(0, len(X), batch_size)
        ])
