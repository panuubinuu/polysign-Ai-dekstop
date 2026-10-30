"""PolySign AI - Native Windows Desktop Application
High-performance, zero-glitch, offline-capable Indian Sign Language Recognition & Translation.
Features 20 gestures with high accuracy & native multilingual audio.
"""
import io, json, os, sys, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from polysign_utils import (
    make_hands, frame_features, HAND_CONNECTIONS, SEQ_LEN, classify_hybrid_gesture
)
import audio_manager

LANGS = {
    "मराठी (Marathi)": "mr",
    "हिंदी (Hindi)": "hi",
    "English": "en",
    "தமிழ் (Tamil)": "ta",
    "اردو (Urdu)": "ur"
}

class PolySignDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PolySign AI - Sign Language Application 🤟 (20 Signs)")
        self.root.geometry("1120x740")
        self.root.minsize(980, 660)
        self.root.configure(bg="#0f172a")

        # Load assets
        self.load_assets()

        # State
        self.lang_key = "mr"
        self.running = True
        self.sentence = []
        self.history = []
        self.cap = None
        self.hands = make_hands()
        self.buf = deque(maxlen=SEQ_LEN)
        self.votes = deque(maxlen=3)
        self.armed = True
        self.missing = 0
        self.last_cid = None

        # Build UI
        self.build_ui()

        # Precache vocabulary audio in background
        audio_manager.precache_vocabulary(self.labels)

        # Start Camera
        self.start_camera()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load_assets(self):
        labels_file = BASE_DIR / "multilingual_labels.json"
        model_file = BASE_DIR / "models" / "polysign.keras"
        classes_file = BASE_DIR / "models" / "classes.json"
        tmpl_file = BASE_DIR / "models" / "avatar_templates.npy"

        if not model_file.exists():
            messagebox.showerror("Error", "Model not found. Please train model first.")
            sys.exit(1)

        self.labels = json.load(open(labels_file, encoding="utf-8"))
        self.model = tf.keras.models.load_model(model_file)
        self.classes = json.load(open(classes_file))
        self.templates = np.load(tmpl_file)

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1e1b4b", height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(header, text="🤟 PolySign AI", font=("Segoe UI", 20, "bold"), fg="#38bdf8", bg="#1e1b4b")
        title.pack(side=tk.LEFT, padx=20, pady=10)

        subtitle = tk.Label(header, text="20 Indian Sign Language Gestures | Real-Time Voice & Translation",
                            font=("Segoe UI", 11), fg="#cbd5e1", bg="#1e1b4b")
        subtitle.pack(side=tk.LEFT, padx=5, pady=15)

        # Main Container
        main_frame = tk.Frame(self.root, bg="#0f172a")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Video Feed
        left_col = tk.Frame(main_frame, bg="#1e293b", bd=2, relief=tk.FLAT)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        v_title = tk.Label(left_col, text="📹 Live Camera & Landmark Tracking",
                           font=("Segoe UI", 12, "bold"), fg="#93c5fd", bg="#1e293b")
        v_title.pack(anchor=tk.W, padx=12, pady=(10, 5))

        self.video_canvas = tk.Label(left_col, bg="#020617")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right Column: Detection Info
        right_col = tk.Frame(main_frame, bg="#0f172a", width=380)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_col.pack_propagate(False)

        # SOS Banner
        self.sos_banner = tk.Label(right_col, text="🚨 EMERGENCY SIGN DETECTED!",
                                   font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#b91c1c",
                                   relief=tk.RAISED, bd=2)

        # Live Card
        live_card = tk.LabelFrame(right_col, text="🎯 Detected Sign / ओळखलेले चिन्ह",
                                  font=("Segoe UI", 11, "bold"), fg="#38bdf8", bg="#1e293b",
                                  bd=1, relief=tk.SOLID, padx=15, pady=15)
        live_card.pack(fill=tk.X, pady=(0, 10))

        self.detected_label = tk.Label(live_card, text="✋ …", font=("Segoe UI", 24, "bold"),
                                       fg="#ffffff", bg="#1e293b")
        self.detected_label.pack(pady=5)

        self.conf_label = tk.Label(live_card, text="Confidence: 0%", font=("Segoe UI", 10),
                                   fg="#94a3b8", bg="#1e293b")
        self.conf_label.pack()

        # Sentence Card
        sent_card = tk.LabelFrame(right_col, text="📝 Sentence Builder / संपूर्ण वाक्य",
                                  font=("Segoe UI", 11, "bold"), fg="#a78bfa", bg="#1e293b",
                                  bd=1, relief=tk.SOLID, padx=15, pady=12)
        sent_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.sent_box = tk.Text(sent_card, font=("Segoe UI", 14), fg="#ffffff", bg="#0f172a",
                                height=5, wrap=tk.WORD, bd=0, relief=tk.FLAT)
        self.sent_box.pack(fill=tk.BOTH, expand=True, pady=5)
        self.sent_box.insert(tk.END, "—")
        self.sent_box.configure(state=tk.DISABLED)

        btn_bar = tk.Frame(sent_card, bg="#1e293b")
        btn_bar.pack(fill=tk.X, pady=(5, 0))

        clr_btn = tk.Button(btn_bar, text="🧹 Clear", font=("Segoe UI", 9, "bold"),
                            bg="#475569", fg="#ffffff", activebackground="#64748b",
                            bd=0, padx=10, pady=4, cursor="hand2", command=self.clear_sentence)
        clr_btn.pack(side=tk.LEFT)

        speak_btn = tk.Button(btn_bar, text="🔊 Speak Sentence", font=("Segoe UI", 9, "bold"),
                              bg="#2563eb", fg="#ffffff", activebackground="#3b82f6",
                              bd=0, padx=10, pady=4, cursor="hand2", command=self.speak_sentence)
        speak_btn.pack(side=tk.LEFT, padx=8)

        # Settings Panel
        settings_card = tk.LabelFrame(right_col, text="⚙️ Language & Controls",
                                      font=("Segoe UI", 10, "bold"), fg="#f472b6", bg="#1e293b",
                                      bd=1, relief=tk.SOLID, padx=12, pady=10)
        settings_card.pack(fill=tk.X, pady=(0, 5))

        lang_row = tk.Frame(settings_card, bg="#1e293b")
        lang_row.pack(fill=tk.X, pady=3)

        tk.Label(lang_row, text="Voice Language:", font=("Segoe UI", 9), fg="#e2e8f0", bg="#1e293b").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="मराठी (Marathi)")
        lang_menu = ttk.Combobox(lang_row, textvariable=self.lang_var, values=list(LANGS.keys()),
                                 state="readonly", width=18)
        lang_menu.pack(side=tk.RIGHT)
        lang_menu.bind("<<ComboboxSelected>>", self.on_lang_change)

        self.voice_var = tk.BooleanVar(value=True)
        v_check = tk.Checkbutton(settings_card, text="🔊 Auto Speak Gestures in Selected Language", variable=self.voice_var,
                                 font=("Segoe UI", 9), fg="#e2e8f0", bg="#1e293b", selectcolor="#0f172a",
                                 activebackground="#1e293b", activeforeground="#ffffff")
        v_check.pack(anchor=tk.W, pady=4)

        avatar_btn = tk.Button(settings_card, text="🧑‍🎤 Open Sign Avatar Library (20 Signs)",
                               font=("Segoe UI", 10, "bold"), bg="#8b5cf6", fg="#ffffff",
                               activebackground="#a78bfa", bd=0, pady=6, cursor="hand2",
                               command=self.open_avatar_window)
        avatar_btn.pack(fill=tk.X, pady=(5, 0))

    def on_lang_change(self, event=None):
        name = self.lang_var.get()
        self.lang_key = LANGS.get(name, "mr")
        self.update_sentence_display()

    def clear_sentence(self):
        self.sentence = []
        self.update_sentence_display()

    def update_sentence_display(self):
        words = [self.labels[str(c)][self.lang_key] for c in self.sentence if str(c) in self.labels]
        text = " ".join(words) if words else "—"
        self.sent_box.configure(state=tk.NORMAL)
        self.sent_box.delete("1.0", tk.END)
        self.sent_box.insert(tk.END, text)
        self.sent_box.configure(state=tk.DISABLED)

    def speak_text(self, text):
        audio_manager.speak(text, self.lang_key)

    def speak_sentence(self):
        words = [self.labels[str(c)][self.lang_key] for c in self.sentence if str(c) in self.labels]
        if words:
            self.speak_text(" ".join(words))

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                live_text = "✋ …"
                conf_pct = 0

                if res.multi_hand_landmarks:
                    self.missing = 0
                    for lm in res.multi_hand_landmarks:
                        h, w = frame.shape[:2]
                        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                        for a, b in HAND_CONNECTIONS:
                            if a < len(pts) and b < len(pts):
                                cv2.line(frame, pts[a], pts[b], (248, 189, 56), 2)
                        for x, y in pts:
                            cv2.circle(frame, (x, y), 3, (182, 114, 244), -1)

                    self.buf.append(frame_features(res))

                    # Instant hybrid classification
                    cid, conf = classify_hybrid_gesture(res, self.buf, self.model, self.classes)

                    if cid is not None and conf >= 0.70:
                        info = self.labels[str(cid)]
                        live_text = f"{info[self.lang_key]} ({info['en']})"
                        conf_pct = int(conf * 100)
                        self.votes.append(cid)

                        if len(self.votes) >= 3 and len(set(self.votes)) == 1:
                            if self.armed or self.last_cid != cid:
                                self.sentence.append(cid)
                                self.armed = False
                                self.last_cid = cid
                                self.update_sentence_display()

                                if info.get("emergency"):
                                    sos_msg = f"🚨 EMERGENCY: {info['mr']} / {info['hi']} / {info['en']}"
                                    self.sos_banner.config(text=sos_msg)
                                    self.sos_banner.pack(fill=tk.X, pady=(0, 10), before=self.detected_label.master)
                                else:
                                    self.sos_banner.pack_forget()

                                if self.voice_var.get():
                                    self.speak_text(info[self.lang_key])
                    else:
                        self.votes.append(-1)
                else:
                    self.missing += 1
                    if self.missing > 6:
                        self.buf.clear()
                        self.votes.clear()
                        self.armed = True
                        self.last_cid = None
                        self.sos_banner.pack_forget()

                self.detected_label.config(text=live_text)
                self.conf_label.config(text=f"Confidence: {conf_pct}%" if conf_pct > 0 else "Ready - Show hand")

                # Convert frame for Tkinter canvas
                h, w = frame.shape[:2]
                disp_w = 640
                disp_h = int(h * (disp_w / w))
                frame_resized = cv2.resize(frame, (disp_w, disp_h))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.video_canvas.imgtk = imgtk
                self.video_canvas.configure(image=imgtk)

        self.root.after(25, self.update_frame)

    def open_avatar_window(self):
        win = tk.Toplevel(self.root)
        win.title("🧑‍🎤 Sign Avatar Library - 20 Signs")
        win.geometry("560x540")
        win.configure(bg="#0f172a")

        lbl = tk.Label(win, text="Sign Avatar Visualizer", font=("Segoe UI", 14, "bold"),
                       fg="#38bdf8", bg="#0f172a")
        lbl.pack(pady=10)

        options = [f"{v['en']} | {v[self.lang_key]}" for k, v in self.labels.items() if int(k) in self.classes]
        sel_var = tk.StringVar(value=options[0])
        cb = ttk.Combobox(win, textvariable=sel_var, values=options, state="readonly", width=34)
        cb.pack(pady=5)

        canvas = tk.Label(win, bg="#020617")
        canvas.pack(padx=20, pady=10)

        anim_state = {"idx": 0, "running": True}

        def update_avatar_frame():
            if not anim_state["running"]:
                return
            sel_text = sel_var.get()
            cid = 0
            for k, v in self.labels.items():
                if f"{v['en']} | {v[self.lang_key]}" == sel_text:
                    cid = int(k)
                    break
            c_idx = self.classes.index(cid) if cid in self.classes else 0
            tmpl = self.templates[c_idx]

            W, H, S = 440, 320, 110
            centers = [(130, 210), (310, 210)]
            frame_idx = anim_state["idx"] % SEQ_LEN
            f = tmpl[frame_idx]

            im = Image.new("RGB", (W, H), (15, 23, 42))
            d = ImageDraw.Draw(im)
            for h, (cx, cy) in enumerate(centers):
                p = f[h * 63:(h + 1) * 63].reshape(21, 3)
                if not p.any():
                    continue
                xy = [(cx + a * S, cy + b * S) for a, b, _ in p]
                for conn in HAND_CONNECTIONS:
                    a, b = int(conn[0]), int(conn[1])
                    if a < len(xy) and b < len(xy):
                        d.line([xy[a], xy[b]], fill=(56, 189, 248), width=3)
                for x, y in xy:
                    d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(244, 114, 182))

            imgtk = ImageTk.PhotoImage(image=im)
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)
            anim_state["idx"] += 1
            win.after(70, update_avatar_frame)

        def on_avatar_close():
            anim_state["running"] = False
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_avatar_close)
        update_avatar_frame()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PolySignDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
