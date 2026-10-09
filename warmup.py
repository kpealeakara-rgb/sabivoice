"""Load every model and run one dummy request before the public link opens,
so the first real tester does not wait for model downloads and GPU warm-up.
Nothing here is logged as an interaction."""
import time
import traceback

import numpy as np


def run():
    t0 = time.time()
    import rag, llm, asr, tts

    print("Warming up retrieval...", flush=True)
    rag.search("How do I check my NIN?", k=1)

    print("Warming up speech recognition...", flush=True)
    try:
        asr._run(np.zeros(16000, dtype=np.float32))
        print("Voice OK.", flush=True)
    except Exception:
        print("VOICE PROBLEM at startup (screenshot this):", flush=True)
        traceback.print_exc()

    print("Warming up N-ATLaS (first start downloads the model, about 5 to 10 minutes)...", flush=True)
    llm.chat([
        {"role": "system", "content": "Reply in one short sentence."},
        {"role": "user", "content": "Say hello."},
    ])

    tts.speak("Hello")
    print(f"Warm-up done in {time.time() - t0:.0f}s. Starting the public link...", flush=True)


if __name__ == "__main__":
    run()
