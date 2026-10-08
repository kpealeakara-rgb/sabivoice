"""Speech-to-text with NCAIR's Nigerian-accented English Whisper model."""
import os
import functools
import numpy as np

ASR_MODEL = os.getenv("ASR_MODEL", "NCAIR1/NigerianAccentedEnglish")
TARGET_SR = 16000


@functools.lru_cache(maxsize=1)
def _pipe():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        token=os.getenv("HF_TOKEN"),
        device=device,
        chunk_length_s=30,
    )


def transcribe(audio) -> str:
    """audio: filepath, or (sample_rate, np.ndarray) tuple from gr.Audio."""
    import librosa

    if audio is None:
        return ""
    if isinstance(audio, str):
        wav, _ = librosa.load(audio, sr=TARGET_SR, mono=True)
    else:
        sr, wav = audio
        wav = np.asarray(wav)
        if wav.dtype.kind == "i":
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    if wav.size < TARGET_SR * 0.3:  # under 0.3 s: nothing useful
        return ""
    out = _pipe()({"raw": wav, "sampling_rate": TARGET_SR},
                  generate_kwargs={"language": "english", "task": "transcribe"})
    return (out.get("text") or "").strip()
