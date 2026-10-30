"""Train the GRU sign classifier + build avatar templates.
Usage: python train_model.py
Outputs: models/polysign.keras, models/classes.json, models/avatar_templates.npy
"""
import json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from polysign_utils import SEQ_LEN, FEAT_DIM

rng = np.random.default_rng(42)
data_dir = BASE_DIR / "data"
dirs = sorted(data_dir.glob("class_*"), key=lambda p: int(p.name.split("_")[1]))
class_ids = [int(d.name.split("_")[1]) for d in dirs if list(d.glob("*.npy"))]
if len(class_ids) < 2:
    raise SystemExit("Collect data for at least 2 classes first (collect_landmarks.py or generate_dataset.py).")

X, y = [], []
for idx, cid in enumerate(class_ids):
    for f in (data_dir / f"class_{cid}").glob("*.npy"):
        X.append(np.load(f))
        y.append(idx)
X, y = np.array(X, np.float32), np.array(y)
print("Dataset:", X.shape, "classes:", len(class_ids))

def augment(x):
    """noise + scale + small rotation (only on non-zero landmarks)."""
    mask = (x != 0).astype(np.float32)
    x = x + rng.normal(0, 0.01, x.shape).astype(np.float32) * mask
    x = x * rng.uniform(0.9, 1.1)
    a = np.deg2rad(rng.uniform(-10, 10))
    r = x.reshape(SEQ_LEN, 2, 21, 3).copy()
    c, s = np.cos(a), np.sin(a)
    rx, ry = r[..., 0].copy(), r[..., 1].copy()
    r[..., 0], r[..., 1] = c * rx - s * ry, s * rx + c * ry
    return r.reshape(SEQ_LEN, FEAT_DIM)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=42)
Xaug = np.array([augment(x) for x in Xtr for _ in range(4)], np.float32)
yaug = np.repeat(ytr, 4)
Xtr, ytr = np.concatenate([Xtr, Xaug]), np.concatenate([ytr, yaug])

n = len(class_ids)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, FEAT_DIM)),
    tf.keras.layers.GRU(96, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=120, batch_size=32, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                                     restore_best_weights=True)])

pred = model.predict(Xte, verbose=0).argmax(1)
acc = float((pred == yte).mean())
print(f"\nTEST ACCURACY: {acc*100:.2f}%")
print(classification_report(yte, pred, target_names=[str(c) for c in class_ids], zero_division=0))

models_dir = BASE_DIR / "models"
models_dir.mkdir(exist_ok=True)
model.save(models_dir / "polysign.keras")
json.dump(class_ids, open(models_dir / "classes.json", "w"))

# Avatar templates: the recorded sample closest to each class mean
tmpl = []
for idx in range(n):
    S = X[y == idx]
    m = S.mean(0)
    tmpl.append(S[np.argmin([np.linalg.norm(s - m) for s in S])])
np.save(models_dir / "avatar_templates.npy", np.array(tmpl, np.float32))
print("Saved model + avatar templates to models/ folder.")
