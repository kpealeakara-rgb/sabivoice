"""Speech-to-text with NCAIR's Nigerian-accented English Whisper model."""
import os
import functools
import traceback

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


def _run(wav: np.ndarray) -> str:
    pipe = _pipe()
    sample = {"raw": wav, "sampling_rate": TARGET_SR}
    try:
        out = pipe(sample, generate_kwargs={"language": "english", "task": "transcribe"})
    except Exception as e:
        # Some fine-tuned Whisper checkpoints reject the language hint; try plain.
        print("ASR with language hint failed, retrying without it:", repr(e), flush=True)
        out = pipe({"raw": wav, "sampling_rate": TARGET_SR})
    return (out.get("text") or "").strip()


def _load(audio) -> np.ndarray:
    import librosa

    if isinstance(audio, str):
        try:
            import soundfile as sf

            wav, sr = sf.read(audio, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            return wav.astype(np.float32)
        except Exception:
            wav, _ = librosa.load(audio, sr=TARGET_SR, mono=True)
            return wav.astype(np.float32)
    sr, wav = audio
    wav = np.asarray(wav)
    if wav.dtype.kind == "i":
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    return wav


def transcribe(audio) -> str:
    """audio: filepath, or (sample_rate, np.ndarray) tuple from gr.Audio."""
    if audio is None:
        return ""
    try:
        wav = _load(audio)
        if wav.size < TARGET_SR * 0.3:  # under 0.3 s: nothing useful
            return ""
        return _run(wav)
    except Exception as e:
        print("VOICE PROBLEM during a question:", flush=True)
        traceback.print_exc()
        import gradio as gr

        raise gr.Error(
            "Sorry, I couldn't process your voice note this time. "
            "Please try recording again, or type your question."
        ) from e
