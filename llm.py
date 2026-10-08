"""N-ATLaS answer generation. CPU (GGUF via llama.cpp) or GPU (transformers)."""
import os
import datetime
import functools

BACKEND = os.getenv("LLM_BACKEND", "llamacpp")
HF_MODEL = os.getenv("LLM_MODEL", "NCAIR1/N-ATLaS")
GGUF_REPO = os.getenv("GGUF_REPO", "tosinamuda/N-ATLaS-GGUF")
GGUF_FILE = os.getenv("GGUF_FILE", "*Q4_K_M.gguf")
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "320"))


@functools.lru_cache(maxsize=1)
def _llamacpp():
    from llama_cpp import Llama

    return Llama.from_pretrained(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        n_ctx=4096,
        n_threads=os.cpu_count() or 2,
        chat_format="llama-3",
        verbose=False,
    )


@functools.lru_cache(maxsize=1)
def _hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tok, model


def chat(messages: list[dict]) -> str:
    if BACKEND == "transformers":
        import torch

        tok, model = _hf()
        date_string = datetime.date.today().strftime("%d %b %Y")
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, date_string=date_string
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.2,
                top_p=0.9, repetition_penalty=1.1,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    llm = _llamacpp()
    res = llm.create_chat_completion(
        messages=messages, max_tokens=MAX_NEW, temperature=0.2, top_p=0.9,
        repeat_penalty=1.1,
    )
    return res["choices"][0]["message"]["content"].strip()
