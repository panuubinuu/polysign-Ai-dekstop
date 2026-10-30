"""Script A - collect dataset.
Usage:  python collect_landmarks.py --class_id 0 --samples 40
Controls: SPACE = record one sample (30 frames), Q = quit.
Saves:   data/class_<id>/<n>.npy   (shape 30 x 126)
Tip: record with different people, lighting, backgrounds and distances for 95%+ real-world accuracy.
"""
import argparse, json, time, sys
from pathlib import Path
import cv2, numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from polysign_utils import make_hands, frame_features, mp_draw, HAND_CONNECTIONS, SEQ_LEN

ap = argparse.ArgumentParser()
ap.add_argument("--class_id", type=int, required=True)
ap.add_argument("--samples", type=int, default=40)
args = ap.parse_args()

labels = json.load(open(BASE_DIR / "multilingual_labels.json", encoding="utf-8"))
name = labels[str(args.class_id)]["en"]
out = BASE_DIR / "data" / f"class_{args.class_id}"
out.mkdir(parents=True, exist_ok=True)
count = len(list(out.glob("*.npy")))

cap = cv2.VideoCapture(0)
hands = make_hands()
state, t0, seq, misses = "idle", 0.0, [], 0

while count < args.samples:
    ok, frame = cap.read()
    if not ok:
        print("Camera not found or not accessible.")
        break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, HAND_CONNECTIONS)

    msg = f"'{name}'  {count}/{args.samples}  - SPACE to record"
    if state == "countdown":
        left = 1.5 - (time.time() - t0)
        msg = f"Get ready... {max(left, 0):.1f}"
        if left <= 0:
            state, seq, misses = "recording", [], 0
    elif state == "recording":
        if res.multi_hand_landmarks:
            seq.append(frame_features(res))
        else:
            misses += 1
        msg = f"RECORDING {len(seq)}/{SEQ_LEN}"
        if misses > 10:
            state = "idle"; print("Hand lost - sample discarded")
        elif len(seq) == SEQ_LEN:
            np.save(out / f"{count}.npy", np.array(seq, np.float32))
            count += 1; state = "idle"

    cv2.putText(frame, msg, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("PolySign - Data Collection", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == 32 and state == "idle":
        state, t0 = "countdown", time.time()

cap.release(); cv2.destroyAllWindows()
print(f"Saved {count} samples for '{name}' in {out}")
