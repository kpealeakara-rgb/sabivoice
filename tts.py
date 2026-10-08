"""Spoken reply. gTTS with a Nigerian English voice where available."""
import tempfile


def speak(text: str):
    if not text:
        return None
    try:
        from gtts import gTTS

        try:
            t = gTTS(text=text, lang="en", tld="com.ng")
        except Exception:
            t = gTTS(text=text, lang="en")
        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        t.save(f.name)
        return f.name
    except Exception as e:
        print("tts failed:", e)
        return None
