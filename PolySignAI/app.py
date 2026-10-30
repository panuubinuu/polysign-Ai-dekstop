"""PolySign AI - Streamlit app.   Run:  streamlit run app.py"""
import io, json, time, threading, base64
from collections import deque
from datetime import datetime
from pathlib import Path

import av, cv2, numpy as np, streamlit as st
from gtts import gTTS
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

BASE_DIR = Path(__file__).resolve().parent

from polysign_utils import (
    make_hands, frame_features, mp_draw, HAND_CONNECTIONS, SEQ_LEN, classify_hybrid_gesture
)

st.set_page_config(page_title="PolySign AI", page_icon="🤟", layout="wide")

LANGS = {
    "मराठी (Marathi)": "mr",
    "हिंदी (Hindi)": "hi",
    "English": "en",
    "தமிழ் (Tamil)": "ta",
    "اردو (Urdu)": "ur"
}

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);color:#e2e8f0}
.hero{font-size:2.5rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem}
.card{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);border-radius:18px;
 padding:18px;margin-bottom:12px}
.big{font-size:2.2rem;font-weight:700;color:#fff}
.sos{background:#b91c1c;border-radius:14px;padding:14px;font-size:1.3rem;font-weight:700;
 text-align:center;animation:p 1s infinite;margin-bottom:12px;color:#fff}
@keyframes p{50%{opacity:.6}}
</style>""", unsafe_allow_html=True)

# ---------- load assets ----------
@st.cache_resource
def load_assets():
    import tensorflow as tf
    labels_file = BASE_DIR / "multilingual_labels.json"
    model_file = BASE_DIR / "models" / "polysign.keras"
    classes_file = BASE_DIR / "models" / "classes.json"
    tmpl_file = BASE_DIR / "models" / "avatar_templates.npy"

    labels = json.load(open(labels_file, encoding="utf-8"))
    model = tf.keras.models.load_model(model_file)
    classes = json.load(open(classes_file))
    tmpl = np.load(tmpl_file)
    return labels, model, classes, tmpl

if not (BASE_DIR / "models" / "polysign.keras").exists():
    st.error("Model not found. Run python generate_dataset.py or train_model.py first.")
    st.stop()

LABELS, MODEL, CLASSES, TEMPLATES = load_assets()

@st.cache_data(show_spinner=False)
def tts_bytes(text, lang):
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()

def speak(text, lang, slot):
    try:
        data = tts_bytes(text, lang)
        b64 = base64.b64encode(data).decode()
        slot.markdown(f"""
        <audio autoplay="true" style="display:none">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)
    except Exception:
        # Fallback to browser SpeechSynthesis in respective language
        lang_code = f"{lang}-IN" if lang in ["mr", "hi", "ta"] else ("ur-PK" if lang == "ur" else "en-US")
        slot.markdown(f"""
        <script>
        try {{
            let u = new SpeechSynthesisUtterance({json.dumps(text)});
            u.lang = "{lang_code}";
            window.speechSynthesis.speak(u);
        }} catch(e) {{}}
        </script>
        """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def avatar_gif(idx):
    """Skeleton 'sign avatar': replays the recorded gesture as an animated pair of hands."""
    W, H, S = 480, 360, 130
    centers = [(150, 230), (330, 230)]
    frames = []
    for f in TEMPLATES[idx]:
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
                    d.line([xy[a], xy[b]], fill=(56, 189, 248), width=4)
            for x, y in xy:
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(244, 114, 182))
        frames.append(im)
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=70, loop=0)
    return out.getvalue()

# ---------- real-time processor ----------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = make_hands()
        self.buf = deque(maxlen=SEQ_LEN)
        self.votes = deque(maxlen=3)
        self.lock = threading.Lock()
        self.events, self.live = [], (None, 0.0)
        self.missing = 0
        self.armed = True
        self.last_cid = None

    def pop_events(self):
        with self.lock:
            e, self.events = self.events, []
            return e, self.live

    def recv(self, frame):
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        res = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, lm, HAND_CONNECTIONS)
            self.missing = 0
            self.buf.append(frame_features(res))

            cid, conf = classify_hybrid_gesture(res, self.buf, MODEL, CLASSES)

            with self.lock:
                self.live = (cid if cid is not None and conf >= 0.70 else None, conf)

            if cid is not None and conf >= 0.70:
                self.votes.append(cid)
                if len(self.votes) >= 3 and len(set(self.votes)) == 1:
                    with self.lock:
                        if self.armed or self.last_cid != cid:
                            self.armed = False
                            self.last_cid = cid
                            self.events.append((cid, conf))
            else:
                self.votes.append(-1)
        else:
            self.missing += 1
            if self.missing > 6:
                self.buf.clear()
                self.votes.clear()
                self.armed = True
                self.last_cid = None
                with self.lock:
                    self.live = (None, 0.0)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- state ----------
ss = st.session_state
ss.setdefault("sentence", [])
ss.setdefault("history", [])

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    lang_name = st.selectbox("Output language / भाषा निवडा", list(LANGS))
    lang = LANGS[lang_name]
    auto_speak = st.toggle("🔊 Auto speak / आवाज सुरू", True)
    cam_mode = st.radio("Camera Source", ["Browser WebRTC", "Local Direct Webcam"], index=0)
    st.markdown("---")
    if st.button("🧹 Clear sentence / वाक्य साफ करा"):
        ss.sentence = []
    if ss.history:
        txt = "\n".join(f"{t} | {w}" for t, w in ss.history)
        st.download_button("💾 Download conversation", txt, "polysign_chat.txt")
    st.caption("💡 Tip: २० भारतीय सांकेतिक भाषा चिन्हे उपलब्ध आहेत.")

st.markdown('<div class="hero">🤟 PolySign AI</div>', unsafe_allow_html=True)
st.caption("20 Indian Sign Language Gestures | Real-Time Voice & Translation (मराठी, हिंदी, English, தமிழ், اردو)")

tab1, tab2 = st.tabs(["📷 Sign → Text & Voice", "🧑‍🎤 Text → Sign Avatar"])

with tab1:
    left, right = st.columns([3, 2])
    with right:
        live_slot = st.empty()
        sos_slot = st.empty()
        sent_slot = st.empty()
        audio_slot = st.empty()

    def render_panel(live):
        cid, conf = live
        if cid is not None and str(cid) in LABELS:
            detected_word = LABELS[str(cid)][lang]
            en_word = LABELS[str(cid)]["en"]
            live_slot.markdown(
                f'<div class="card">Detecting / ओळखले<div class="big">{detected_word} ({en_word})</div>'
                f'Confidence {conf*100:.0f}%</div>', unsafe_allow_html=True)
        else:
            live_slot.markdown('<div class="card">Detecting / शोधत आहे<div class="big">✋ …</div>'
                               'कॅमेरासमोर हात दाखवा (Show a sign to camera)</div>', unsafe_allow_html=True)
        words = " ".join(LABELS[str(c)][lang] for c in ss.sentence if str(c) in LABELS) or "—"
        sent_slot.markdown(f'<div class="card">📝 Sentence / संपूर्ण वाक्य<div class="big" '
                           f'dir="auto">{words}</div></div>', unsafe_allow_html=True)

    render_panel((None, 0.0))

    if cam_mode == "Browser WebRTC":
        with left:
            ctx = webrtc_streamer(
                key="polysign", video_processor_factory=SignProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False}, async_processing=True)

        while ctx.state.playing:
            proc = ctx.video_processor
            if proc:
                events, live = proc.pop_events()
                for cid, conf in events:
                    if str(cid) in LABELS:
                        info = LABELS[str(cid)]
                        ss.sentence.append(cid)
                        ss.history.append((datetime.now().strftime("%H:%M:%S"), info["en"]))
                        if info.get("emergency"):
                            sos_slot.markdown(
                                '<div class="sos">🚨 EMERGENCY / तातडीची मदत: ' +
                                " | ".join(info[k] for k in ("mr", "hi", "en", "ta", "ur")) +
                                "</div>", unsafe_allow_html=True)
                        else:
                            sos_slot.empty()
                        if auto_speak:
                            speak(info[lang], lang, audio_slot)
                render_panel(live)
            time.sleep(0.08)
    else:
        # Direct Webcam via OpenCV
        with left:
            start_cam = st.toggle("🎥 Start Direct Camera", False)
            frame_slot = st.empty()

        if start_cam:
            cap = cv2.VideoCapture(0)
            hands = make_hands()
            buf = deque(maxlen=SEQ_LEN)
            votes = deque(maxlen=3)
            armed = True
            missing = 0
            last_cid = None

            while start_cam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot access local camera.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                live_cid = None
                conf = 0.0

                if res.multi_hand_landmarks:
                    missing = 0
                    for lm in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, lm, HAND_CONNECTIONS)
                    buf.append(frame_features(res))

                    cid, conf = classify_hybrid_gesture(res, buf, MODEL, CLASSES)

                    if cid is not None and conf >= 0.70:
                        live_cid = cid
                        votes.append(cid)
                        if len(votes) >= 3 and len(set(votes)) == 1:
                            if armed or last_cid != cid:
                                armed = False
                                last_cid = cid
                                info = LABELS[str(cid)]
                                ss.sentence.append(cid)
                                ss.history.append((datetime.now().strftime("%H:%M:%S"), info["en"]))
                                if info.get("emergency"):
                                    sos_slot.markdown(
                                        '<div class="sos">🚨 EMERGENCY / तातडीची मदत: ' +
                                        " | ".join(info[k] for k in ("mr", "hi", "en", "ta", "ur")) +
                                        "</div>", unsafe_allow_html=True)
                                else:
                                    sos_slot.empty()
                                if auto_speak:
                                    speak(info[lang], lang, audio_slot)
                    else:
                        votes.append(-1)
                else:
                    missing += 1
                    if missing > 6:
                        buf.clear()
                        votes.clear()
                        armed = True
                        last_cid = None

                render_panel((live_cid, conf))
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                time.sleep(0.02)
            cap.release()

with tab2:
    st.write("Type a word (English, मराठी, हिंदी, etc.) or pick a sign to watch the Sign Avatar replay it (20 Signs).")
    c1, c2 = st.columns(2)
    typed = c1.text_input("Type text / शब्द लिहा")
    options = {f'{v["en"]} | {v[lang]}': k for k, v in LABELS.items()
               if int(k) in CLASSES}
    picked = c2.selectbox("…or choose a sign / चिन्ह निवडा", ["—"] + list(options.keys()))

    wanted = []
    if picked != "—":
        wanted.append(options[picked])
    if typed:
        for w in typed.lower().replace(",", " ").split():
            for k, v in LABELS.items():
                vals = {v["gesture_token"].lower(), v["en"].lower(),
                        *(str(v[x]).lower() for x in ("mr", "hi", "ta", "ur"))}
                if w in vals and int(k) in CLASSES and k not in wanted:
                    wanted.append(k)

    if not wanted and typed:
        st.warning("No matching sign found in the dictionary / शब्दकोशात हे चिन्ह सापडले नाही.")
    elif wanted:
        chunk_size = 3
        for i in range(0, len(wanted), chunk_size):
            chunk = wanted[i:i + chunk_size]
            cols = st.columns(len(chunk))
            for col, k in zip(cols, chunk):
                col.image(avatar_gif(CLASSES.index(int(k))), caption=f"{LABELS[k]['en']} / {LABELS[k][lang]}")
                if col.button(f"🔊 {LABELS[k][lang]}", key=f"sp_{k}_{i}"):
                    speak(LABELS[k][lang], lang, col.empty())
