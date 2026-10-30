"""PolySign AI Audio Manager
High-performance, zero-latency, offline-first multilingual speech synthesis.
Supports Marathi (mr), Hindi (hi), English (en), Tamil (ta), Urdu (ur).
"""
import os, sys, threading, hashlib, tempfile, ctypes
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "audio_cache"
CACHE_DIR.mkdir(exist_ok=True)

_play_lock = threading.Lock()
_winmm = ctypes.windll.winmm

def _get_cache_path(text, lang):
    key = f"{lang}_{text.strip()}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    return CACHE_DIR / f"{lang}_{h}.mp3"

def _play_mp3_native(filepath):
    try:
        p = str(filepath).replace("/", "\\")
        with _play_lock:
            _winmm.mciSendStringW("close polysign_voice", None, 0, 0)
            _winmm.mciSendStringW(f'open "{p}" type mpegvideo alias polysign_voice', None, 0, 0)
            _winmm.mciSendStringW("play polysign_voice", None, 0, 0)
    except Exception as e:
        print(f"Audio playback error: {e}")

def speak(text, lang="mr"):
    """Speak text in the specified language (mr, hi, en, ta, ur).
    Non-blocking, cached, and glitch-free.
    """
    if not text or not text.strip():
        return

    def _worker():
        cache_path = _get_cache_path(text, lang)
        if cache_path.exists():
            _play_mp3_native(cache_path)
            return

        # Generate via gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang)
            tts.save(str(cache_path))
            _play_mp3_native(cache_path)
        except Exception:
            # Fallback to Windows SAPI (English / default system voice)
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def precache_vocabulary(labels_dict):
    """Precache audio for all signs in all 5 languages in the background."""
    def _cacher():
        from gtts import gTTS
        langs = ["mr", "hi", "en", "ta", "ur"]
        for cid, info in labels_dict.items():
            for lg in langs:
                word = info.get(lg)
                if word:
                    cp = _get_cache_path(word, lg)
                    if not cp.exists():
                        try:
                            gTTS(text=word, lang=lg).save(str(cp))
                        except Exception:
                            pass
        print(f"Precached {len(list(CACHE_DIR.glob('*.mp3')))} audio clips in audio_cache/")

    t = threading.Thread(target=_cacher, daemon=True)
    t.start()

if __name__ == "__main__":
    import json
    labels = json.load(open(BASE_DIR / "multilingual_labels.json", encoding="utf-8"))
    print("Testing audio manager with Marathi:")
    speak("नमस्कार", "mr")
    import time
    time.sleep(2)
