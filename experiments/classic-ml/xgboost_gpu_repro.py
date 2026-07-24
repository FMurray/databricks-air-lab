"""XGBoost-on-GPU repro (UAT W5) — the known report is "docs notebook hangs on H100, works
on A10". Same script both GPU types; a hang here trips the workload timeout, which IS the
repro receipt (run id + timeout state). Prints heartbeats so the last-logged phase pinpoints
where a hang occurs.
"""
import time

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print(f"RESULT xgboost_version={xgb.__version__}", flush=True)

X, y = make_classification(n_samples=500_000, n_features=100, n_informative=40, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
print("PHASE data_ready", flush=True)

t0 = time.time()
clf = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=200, max_depth=8)
clf.fit(X_tr, y_tr)
fit_s = time.time() - t0
print(f"PHASE fit_done seconds={fit_s:.1f}", flush=True)

auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
assert auc > 0.9, f"suspicious AUC {auc:.3f} — training likely degenerate"
print(f"RESULT xgboost_gpu=PASS fit_seconds={fit_s:.1f} auc={auc:.4f}", flush=True)
