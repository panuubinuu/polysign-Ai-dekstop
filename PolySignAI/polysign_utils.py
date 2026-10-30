"""Shared helpers: landmark extraction + normalization + robust hybrid gesture classification.
Supports 20 Indian Sign Language gestures with instant geometric & neural recognition.
"""
import os, sys, math
from pathlib import Path
import numpy as np
import cv2

SEQ_LEN = 30            # frames per gesture
FEAT_DIM = 126          # 2 hands x 21 landmarks x (x, y, z)

# Standard MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17)                                # palm base
]

class _LandmarkPoint:
    __slots__ = ("x", "y", "z")
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

class _HandLandmarks:
    def __init__(self, points):
        self.landmark = [_LandmarkPoint(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in points]

class _Classification:
    def __init__(self, label):
        self.label = label

class _Handedness:
    def __init__(self, label):
        self.classification = [_Classification(label)]

class _DetectionResult:
    def __init__(self, multi_hand_landmarks=None, multi_handedness=None):
        self.multi_hand_landmarks = multi_hand_landmarks or []
        self.multi_handedness = multi_handedness or []

class _DrawingUtils:
    @staticmethod
    def draw_landmarks(image, hand_landmarks, connections=HAND_CONNECTIONS, landmark_drawing_spec=None, connection_drawing_spec=None):
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark"):
            return
        h, w = image.shape[:2]
        pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark]
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(image, pts[a], pts[b], (56, 189, 248), 2)
        for x, y in pts:
            cv2.circle(image, (x, y), 3, (244, 114, 182), -1)

class _ModernHandsDetector:
    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if model_path is None:
            base_dir = Path(__file__).resolve().parent
            candidate = base_dir / "models" / "hand_landmarker.task"
            if not candidate.exists():
                candidate = base_dir.parent / "models" / "hand_landmarker.task"
            model_path = str(candidate)

        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45
        )
        self.detector = vision.HandLandmarker.create_from_options(opts)

    def process(self, rgb_image):
        import mediapipe as mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        res = self.detector.detect(mp_img)
        multi_lms = []
        multi_hds = []
        if res.hand_landmarks:
            for lms, handedness in zip(res.hand_landmarks, res.handedness):
                pts = [[p.x, p.y, p.z] for p in lms]
                multi_lms.append(_HandLandmarks(pts))
                label = handedness[0].category_name if handedness else "Right"
                multi_hds.append(_Handedness(label))
        return _DetectionResult(multi_lms, multi_hds)

# Setup module exports
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        def make_hands():
            return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        class _MpHandsNamespace:
            HAND_CONNECTIONS = HAND_CONNECTIONS
        mp_hands = _MpHandsNamespace()
        mp_draw = _DrawingUtils()
        def make_hands():
            return _ModernHandsDetector()
except Exception:
    class _MpHandsNamespace:
        HAND_CONNECTIONS = HAND_CONNECTIONS
    mp_hands = _MpHandsNamespace()
    mp_draw = _DrawingUtils()
    def make_hands():
        return _ModernHandsDetector()


def normalize_hand(hand_landmarks):
    """Wrist-relative + scale-invariant -> 63 values."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]                                   # wrist = origin
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale                                # distance-from-camera invariant
    return pts.flatten()


def frame_features(results):
    """126-d vector: [left hand 63 | right hand 63];
    Deterministically handles 1 or 2 hands:
    - 1 hand: populated in BOTH left & right to guarantee left/right mirror invariance.
    - 2 hands: sorted deterministically by screen X coordinate.
    """
    left = np.zeros(63, np.float32)
    right = np.zeros(63, np.float32)

    if results and results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks
        if len(lms) == 1:
            feats = normalize_hand(lms[0])
            left = feats
            right = feats
        elif len(lms) >= 2:
            x0 = lms[0].landmark[0].x
            x1 = lms[1].landmark[0].x
            if x0 <= x1:
                left = normalize_hand(lms[0])
                right = normalize_hand(lms[1])
            else:
                left = normalize_hand(lms[1])
                right = normalize_hand(lms[0])

    return np.concatenate([left, right])


def analyze_single_hand(pts_63):
    """Analyzes a 63-d normalized hand:
    Returns dict of geometric features.
    """
    pts = pts_63.reshape(21, 3)
    if not pts.any():
        return {
            "ext": [False]*5, "spread": 0.0, "pinch_d": 1.0, "ok_d": 1.0,
            "is_fist": False, "is_open": False, "is_pinch": False,
            "is_ok": False, "is_ily": False, "is_call": False
        }

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    mcps = [2, 5, 9, 13, 17]

    # Thumb: tip 4 vs index MCP 5 and thumb IP 3
    thumb_up = (pts[4][1] < pts[5][1] - 0.04) and (pts[4][1] < pts[3][1])
    thumb_out = (np.linalg.norm(pts[4] - pts[2]) > 0.42)

    ext = [thumb_up or thumb_out]
    for i in range(1, 5):
        tip_y = pts[tips[i]][1]
        pip_y = pts[pips[i]][1]
        mcp_y = pts[mcps[i]][1]
        d_tip = np.linalg.norm(pts[tips[i]])
        d_pip = np.linalg.norm(pts[pips[i]])
        is_open = (tip_y < pip_y + 0.05) and (tip_y < mcp_y) and (d_tip > d_pip * 1.03)
        ext.append(bool(is_open))

    # Spread between tips
    tip_pts = [pts[t] for t in tips]
    dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            dists.append(np.linalg.norm(tip_pts[i] - tip_pts[j]))
    spread = float(np.mean(dists))

    # Distances
    pinch_d = float(min(np.linalg.norm(pts[4] - pts[8]), np.linalg.norm(pts[4] - pts[12])))
    ok_d = float(np.linalg.norm(pts[4] - pts[8]))

    is_fist = (sum(ext[1:]) == 0)
    is_open = (sum(ext[1:]) >= 3 and spread > 0.28)
    is_pinch = (spread < 0.26 and pinch_d < 0.22)
    is_ok = (ok_d < 0.16 and ext[2] and ext[3])
    is_ily = (ext[0] and ext[1] and ext[4] and not ext[2] and not ext[3])
    is_call = (ext[0] and ext[4] and not ext[1] and not ext[2] and not ext[3])

    return {
        "ext": ext, "spread": spread, "pinch_d": pinch_d, "ok_d": ok_d,
        "is_fist": is_fist, "is_open": is_open, "is_pinch": is_pinch,
        "is_ok": is_ok, "is_ily": is_ily, "is_call": is_call,
        "thumb_up": thumb_up
    }


def classify_hybrid_gesture(results, buf, model, classes):
    """Hybrid Classifier supporting 20 Indian Sign Language gestures:
    Combines instant geometric pose analysis, motion tracking, and neural GRU.
    Returns: (predicted_class_id, confidence) or (None, 0.0)
    """
    if not results or not results.multi_hand_landmarks:
        return None, 0.0

    lms = results.multi_hand_landmarks
    num_hands = len(lms)

    # 1. TWO-HAND GESTURES
    if num_hands >= 2:
        h0 = np.array([[p.x, p.y, p.z] for p in lms[0].landmark], dtype=np.float32)
        h1 = np.array([[p.x, p.y, p.z] for p in lms[1].landmark], dtype=np.float32)

        d_wrists = np.linalg.norm(h0[0] - h1[0])
        d_tips_mid = np.linalg.norm(h0[12] - h1[12])
        d_wrist_01 = np.linalg.norm(h0[8] - h1[0])
        d_wrist_10 = np.linalg.norm(h1[8] - h0[0])

        # Doctor (9): One hand index/middle checks wrist of other hand
        if d_wrist_01 < 0.22 or d_wrist_10 < 0.22:
            return 9, 0.98

        # Danger (16): Wrists crossed in X-shape
        if d_wrists < 0.20 and (h0[12][1] < h0[0][1] and h1[12][1] < h1[0][1]):
            return 16, 0.98

        # Home (17): Fingertips touching at top, wrists apart (roof triangle)
        if d_tips_mid < 0.16 and d_wrists > 0.28:
            return 17, 0.98

        # Medicine (18): One hand flat palm, other hand grinding palm center
        d_palm_01 = np.linalg.norm(h0[12] - h1[9])
        d_palm_10 = np.linalg.norm(h1[12] - h0[9])
        if d_palm_01 < 0.20 or d_palm_10 < 0.20:
            return 18, 0.98

        # Help (2): One hand under other hand
        y_diff = abs(h0[0][1] - h1[0][1])
        if d_wrists < 0.38 and y_diff > 0.05:
            return 2, 0.98

        # Friend (19): Hands linked/clasped
        if d_wrists < 0.30 and d_tips_mid < 0.20:
            return 19, 0.96

    # 2. SINGLE-HAND GESTURES
    active_hand = normalize_hand(lms[0])
    geo = analyze_single_hand(active_hand)
    ext = geo["ext"]

    # Motion features if buffer has history
    dx_var = 0.0
    dy_var = 0.0
    net_dy = 0.0
    if buf and len(buf) >= 6:
        seq = np.array(buf)
        xs = seq[:, 0]
        ys = seq[:, 1]
        dx_var = float(np.var(xs))
        dy_var = float(np.var(ys))
        net_dy = float(ys[-1] - ys[0])

    # 12. I Love You (Class 12: Thumb, Index, Pinky 🤟)
    if geo["is_ily"]:
        return 12, 0.98

    # 13. Call Me (Class 13: Thumb and Pinky 🤙)
    if geo["is_call"]:
        return 13, 0.98

    # 15. OK (Class 15: Thumb and Index ring, 3 fingers up 👌)
    if geo["is_ok"]:
        return 15, 0.98

    # 10. Good (Class 10: Thumbs up 👍)
    if geo["thumb_up"] and geo["is_fist"]:
        return 10, 0.98

    # 8. Food (Class 8: Pinch 🤏)
    if geo["is_pinch"]:
        return 8, 0.96

    # 7. Water (Class 7: 'W' sign - Index, Middle, Ring up)
    if ext[1] and ext[2] and ext[3] and not ext[4]:
        return 7, 0.98

    # 4 & 14. Two fingers up (Index & Middle):
    if ext[1] and ext[2] and not ext[3] and not ext[4]:
        # Horizontal movement/shaking = No (Class 4)
        if dx_var > 0.0006:
            return 4, 0.96
        # Upright steady V = Victory (Class 14)
        return 14, 0.95

    # 3 & 6. Fist (Class 3: Yes / Class 6: Sorry)
    if geo["is_fist"] and not geo["thumb_up"]:
        if dy_var > 0.0006:
            return 3, 0.95 # Nodding = Yes
        if dx_var > 0.0004 and dy_var > 0.0004:
            return 6, 0.94 # Circular = Sorry
        return 3, 0.88     # Default fist = Yes

    # 11, 0, 1, 5. Open Palm gestures:
    if geo["is_open"]:
        # Waving side-to-side = Hello (Class 0)
        if dx_var > 0.0007:
            return 0, 0.96
        # Downward from chin = Thank you (Class 1)
        if net_dy > 0.04 and dy_var > 0.0004:
            return 1, 0.94
        # Circular motion = Please (Class 5)
        if dx_var > 0.0003 and dy_var > 0.0003:
            return 5, 0.93
        # Steady open palm = Stop (Class 11)
        return 11, 0.96

    # 3. NEURAL GRU FALLBACK / REINFORCEMENT
    if buf and len(buf) >= 8 and model:
        raw_seq = np.array(buf)
        if len(raw_seq) < SEQ_LEN:
            pad = np.repeat(raw_seq[-1:], SEQ_LEN - len(raw_seq), axis=0)
            in_seq = np.vstack([raw_seq, pad])
        else:
            in_seq = raw_seq[-SEQ_LEN:]
        
        preds = model(in_seq[np.newaxis, ...], training=False).numpy()[0]
        top_idx = int(preds.argmax())
        top_conf = float(preds[top_idx])
        if top_conf >= 0.65:
            cid = classes[top_idx] if top_idx < len(classes) else top_idx
            return cid, top_conf

    return None, 0.0
