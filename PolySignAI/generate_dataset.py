"""Generate realistic synthetic baseline dataset for all 20 sign classes
and train the GRU model so PolySign AI works out-of-the-box.
"""
import os, json, math
from pathlib import Path
import numpy as np

SEQ_LEN = 30
FEAT_DIM = 126

def get_hand_landmarks(thumb_open=True, index_open=True, middle_open=True, ring_open=True, pinky_open=True, pinch=False, ok_sign=False):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0] # Wrist

    if pinch:
        # All fingertips touch thumb tip in a cone shape (Food gesture)
        pts[1] = [-0.15, -0.2, 0.0]
        pts[2] = [-0.2, -0.35, 0.05]
        pts[3] = [-0.15, -0.5, 0.05]
        pts[4] = [0.0, -0.65, 0.05]

        for start_idx, base in [(5, [-0.1, -0.4, 0.0]), (9, [0.0, -0.45, 0.0]), (13, [0.1, -0.4, 0.0]), (17, [0.2, -0.35, 0.0])]:
            pts[start_idx] = base
            pts[start_idx+1] = [base[0]*0.7, -0.5, 0.02]
            pts[start_idx+2] = [base[0]*0.3, -0.58, 0.04]
            pts[start_idx+3] = [0.0, -0.65, 0.05]

    elif ok_sign:
        # OK gesture: Thumb (4) and Index (8) touch in an O-ring; Middle, Ring, Pinky extended upright
        pts[1] = [-0.2, -0.2, 0.0]
        pts[2] = [-0.3, -0.35, 0.0]
        pts[3] = [-0.25, -0.50, 0.0]
        pts[4] = [-0.15, -0.58, 0.0] # Thumb tip

        # Index curls to meet thumb tip
        pts[5] = [-0.15, -0.5, 0.0]
        pts[6] = [-0.25, -0.65, 0.0]
        pts[7] = [-0.22, -0.62, 0.0]
        pts[8] = [-0.15, -0.58, 0.0] # Index tip meets thumb tip

        # Middle, Ring, Pinky extended upright
        for base, idxs in [([0.05, -0.60], [9, 10, 11, 12]), ([0.22, -0.55], [13, 14, 15, 16]), ([0.35, -0.45], [17, 18, 19, 20])]:
            bx, by = base
            pts[idxs[0]] = [bx, by, 0.0]
            pts[idxs[1]] = [bx * 1.15, by - 0.22, 0.0]
            pts[idxs[2]] = [bx * 1.25, by - 0.42, 0.0]
            pts[idxs[3]] = [bx * 1.35, by - 0.62, 0.0]

    else:
        # Standard finger configuration
        if thumb_open:
            pts[1] = [-0.25, -0.2, 0.0]
            pts[2] = [-0.4, -0.4, 0.0]
            pts[3] = [-0.5, -0.6, 0.0]
            pts[4] = [-0.55, -0.75, 0.0]
        else:
            pts[1] = [-0.2, -0.15, 0.0]
            pts[2] = [-0.25, -0.3, -0.05]
            pts[3] = [-0.15, -0.35, -0.12]
            pts[4] = [-0.05, -0.38, -0.15]

        fingers = [
            (index_open, -0.2, -0.55, [5, 6, 7, 8]),
            (middle_open, 0.0, -0.60, [9, 10, 11, 12]),
            (ring_open, 0.2, -0.55, [13, 14, 15, 16]),
            (pinky_open, 0.35, -0.45, [17, 18, 19, 20])
        ]
        for is_open, bx, by, idxs in fingers:
            pts[idxs[0]] = [bx, by, 0.0]
            if is_open:
                pts[idxs[1]] = [bx * 1.15, by - 0.22, 0.0]
                pts[idxs[2]] = [bx * 1.25, by - 0.42, 0.0]
                pts[idxs[3]] = [bx * 1.35, by - 0.62, 0.0]
            else:
                pts[idxs[1]] = [bx * 0.9, by + 0.12, -0.1]
                pts[idxs[2]] = [bx * 0.7, by + 0.24, -0.15]
                pts[idxs[3]] = [bx * 0.5, by + 0.20, -0.2]

    # Normalize relative to scale
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts /= scale
    return pts

def rotate_hand(pts, yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return pts @ R.T

def generate_sample_for_class(cid, sample_idx):
    rng = np.random.default_rng(cid * 1000 + sample_idx)
    seq = np.zeros((SEQ_LEN, FEAT_DIM), dtype=np.float32)

    phase_offset = rng.uniform(-0.3, 0.3)
    speed_factor = rng.uniform(0.9, 1.1)
    base_jitter = rng.normal(0, 0.008, (SEQ_LEN, 21, 3)).astype(np.float32)

    # Classes requiring 2 hands: 2 (Help), 9 (Doctor), 16 (Danger), 17 (Home), 18 (Medicine), 19 (Friend)
    is_two_handed = (cid in [2, 9, 16, 17, 18, 19])

    for f in range(SEQ_LEN):
        t = (f / float(SEQ_LEN)) * speed_factor + phase_offset
        left_hand = None
        right_hand = None

        if cid == 0:
            # Hello: Open palm waving side to side
            base = get_hand_landmarks(True, True, True, True, True)
            yaw = 0.35 * math.sin(t * math.pi * 3.5)
            roll = 0.20 * math.sin(t * math.pi * 3.5)
            right_hand = rotate_hand(base, yaw=yaw, roll=roll)

        elif cid == 1:
            # Thank you: Flat hand moving forward and down
            base = get_hand_landmarks(True, True, True, True, True)
            pitch = -0.3 + 0.5 * (f / SEQ_LEN)
            yaw = 0.1 * math.sin(t * math.pi)
            right_hand = rotate_hand(base, pitch=pitch, yaw=yaw)

        elif cid == 2:
            # Help: Left open palm base, Right thumbs up lifting
            left_base = get_hand_landmarks(True, True, True, True, True)
            left_hand = rotate_hand(left_base, pitch=1.2, roll=0.2)
            right_base = get_hand_landmarks(thumb_open=True, index_open=False, middle_open=False, ring_open=False, pinky_open=False)
            lift = -0.15 * math.sin(t * math.pi * 2.0)
            right_hand = rotate_hand(right_base, pitch=-0.1 + lift)

        elif cid == 3:
            # Yes: Fist nodding up and down
            base = get_hand_landmarks(False, False, False, False, False)
            pitch = 0.4 * math.sin(t * math.pi * 3.0)
            right_hand = rotate_hand(base, pitch=pitch)

        elif cid == 4:
            # No: Index & middle snapping/shaking
            base = get_hand_landmarks(thumb_open=False, index_open=True, middle_open=True, ring_open=False, pinky_open=False)
            yaw = 0.3 * math.sin(t * math.pi * 4.0)
            right_hand = rotate_hand(base, yaw=yaw)

        elif cid == 5:
            # Please: Flat hand circular motion
            base = get_hand_landmarks(True, True, True, True, True)
            yaw = 0.25 * math.cos(t * math.pi * 2.5)
            pitch = 0.25 * math.sin(t * math.pi * 2.5)
            right_hand = rotate_hand(base, yaw=yaw, pitch=pitch)

        elif cid == 6:
            # Sorry: Fist rubbing circular motion
            base = get_hand_landmarks(False, False, False, False, False)
            yaw = 0.22 * math.cos(t * math.pi * 2.5)
            pitch = 0.22 * math.sin(t * math.pi * 2.5)
            right_hand = rotate_hand(base, yaw=yaw, pitch=pitch)

        elif cid == 7:
            # Water: 'W' sign (Index, Middle, Ring up)
            base = get_hand_landmarks(thumb_open=False, index_open=True, middle_open=True, ring_open=True, pinky_open=False)
            pitch = 0.2 * math.sin(t * math.pi * 3.5)
            right_hand = rotate_hand(base, pitch=pitch)

        elif cid == 8:
            # Food: Pinched fingertips
            base = get_hand_landmarks(pinch=True)
            pitch = 0.15 * math.sin(t * math.pi * 3.0)
            right_hand = rotate_hand(base, pitch=pitch)

        elif cid == 9:
            # Doctor: Left wrist steady, Right 2 fingers tapping left wrist
            left_base = get_hand_landmarks(False, False, False, False, False)
            left_hand = rotate_hand(left_base, pitch=0.8)
            right_base = get_hand_landmarks(thumb_open=False, index_open=True, middle_open=True, ring_open=False, pinky_open=False)
            tap = 0.3 * math.sin(t * math.pi * 3.5)
            right_hand = rotate_hand(right_base, pitch=tap)

        elif cid == 10:
            # Good: Thumbs up
            base = get_hand_landmarks(thumb_open=True, index_open=False, middle_open=False, ring_open=False, pinky_open=False)
            thrust = 0.15 * math.sin(t * math.pi * 2.0)
            right_hand = rotate_hand(base, pitch=thrust, roll=0.1)

        elif cid == 11:
            # Stop: Steady open palm
            base = get_hand_landmarks(True, True, True, True, True)
            shake = 0.04 * math.sin(t * math.pi * 2.0)
            right_hand = rotate_hand(base, yaw=shake, pitch=-0.1)

        elif cid == 12:
            # I Love You: Thumb, Index, Pinky extended 🤟
            base = get_hand_landmarks(thumb_open=True, index_open=True, middle_open=False, ring_open=False, pinky_open=True)
            rock = 0.12 * math.sin(t * math.pi * 2.5)
            right_hand = rotate_hand(base, roll=rock)

        elif cid == 13:
            # Call Me: Thumb and Pinky extended 🤙
            base = get_hand_landmarks(thumb_open=True, index_open=False, middle_open=False, ring_open=False, pinky_open=True)
            shake = 0.25 * math.sin(t * math.pi * 3.0)
            right_hand = rotate_hand(base, yaw=shake, roll=0.2)

        elif cid == 14:
            # Victory: Index and Middle upright in wide V ✌️
            base = get_hand_landmarks(thumb_open=False, index_open=True, middle_open=True, ring_open=False, pinky_open=False)
            thrust = 0.08 * math.sin(t * math.pi * 1.5)
            right_hand = rotate_hand(base, pitch=thrust)

        elif cid == 15:
            # OK: Thumb and Index ring, 3 fingers up 👌
            base = get_hand_landmarks(ok_sign=True)
            thrust = 0.08 * math.sin(t * math.pi * 2.0)
            right_hand = rotate_hand(base, pitch=thrust)

        elif cid == 16:
            # Danger: Two hands crossed at wrists in X shape ⚠️
            h = get_hand_landmarks(True, True, True, True, True)
            left_hand = rotate_hand(h, roll=-0.6, yaw=-0.2)
            right_hand = rotate_hand(h, roll=0.6, yaw=0.2)

        elif cid == 17:
            # Home: Two hands fingertips touching forming a roof 🏠
            h = get_hand_landmarks(True, True, True, True, True)
            left_hand = rotate_hand(h, roll=-0.75, pitch=-0.1)
            right_hand = rotate_hand(h, roll=0.75, pitch=-0.1)

        elif cid == 18:
            # Medicine: Left flat palm, Right middle finger grinding center 💊
            left_base = get_hand_landmarks(True, True, True, True, True)
            left_hand = rotate_hand(left_base, pitch=1.1)
            right_base = get_hand_landmarks(thumb_open=False, index_open=False, middle_open=True, ring_open=False, pinky_open=False)
            grind = 0.2 * math.sin(t * math.pi * 3.5)
            right_hand = rotate_hand(right_base, pitch=grind, yaw=grind)

        elif cid == 19:
            # Friend: Two hands clasped / fingers interlinked 🤝
            h = get_hand_landmarks(thumb_open=True, index_open=True, middle_open=False, ring_open=False, pinky_open=False)
            left_hand = rotate_hand(h, roll=0.4, pitch=0.3)
            right_hand = rotate_hand(h, roll=-0.4, pitch=0.3)

        if is_two_handed:
            left_hand = left_hand + base_jitter[f] * 0.5
            s = np.max(np.linalg.norm(left_hand, axis=1))
            if s > 0: left_hand /= s
            left_feat = left_hand.flatten()

            right_hand = right_hand + base_jitter[f]
            s = np.max(np.linalg.norm(right_hand, axis=1))
            if s > 0: right_hand /= s
            right_feat = right_hand.flatten()
        else:
            right_hand = right_hand + base_jitter[f]
            s = np.max(np.linalg.norm(right_hand, axis=1))
            if s > 0: right_hand /= s
            feat = right_hand.flatten()
            left_feat = feat
            right_feat = feat

        seq[f] = np.concatenate([left_feat, right_feat])

    return seq

def generate_all():
    print("Generating comprehensive dataset for 20 sign classes...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for cid in range(20):
        c_dir = data_dir / f"class_{cid}"
        c_dir.mkdir(exist_ok=True)
        for s in range(50):
            seq = generate_sample_for_class(cid, s)
            np.save(c_dir / f"{s}.npy", seq)
        print(f"  Class {cid:2d}: saved 50 samples in {c_dir}")
    print("Expanded 20-class dataset generation complete!")

if __name__ == "__main__":
    generate_all()
