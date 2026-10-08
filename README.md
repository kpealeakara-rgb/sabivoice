---
title: SabiVoice
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: other
---

# SabiVoice: Nigeria's voice-first citizen helpline

SabiVoice lets any Nigerian ask a question **by voice, in Nigerian-accented English**, about the public services that affect daily life (tax, NIN and BVN, health insurance, pensions, consumer rights, and scam/fraud reporting) and get a short, plain answer grounded in official documents, with the source cited and the right agency to contact next.

Built on **N-ATLaS** (NCAIR / Awarri) for the National AI Innovation Challenge 2027, Problem Statement 2: Voice-First Access.

## How it works

```
voice note ──► NCAIR1/NigerianAccentedEnglish (ASR, Whisper-small fine-tune)
           ──► retrieval over a curated knowledge base of official Nigerian sources
           ──► NCAIR1/N-ATLaS (answer, grounded + cited, Nigerian English)
           ──► spoken reply + text + next step / agency contact
           ──► anonymised log ──► agency dashboard (what citizens are confused about)
```

1. **Speech in** – Nigerian-accented English ASR from NCAIR (`NCAIR1/NigerianAccentedEnglish`).
2. **Grounding** – every answer is retrieved from `kb/`, a curated set of passages from official sources (FIRS/NRS, NIMC, CBN/NIBSS, NHIA, PenCom, FCCPC, NCC, EFCC). Each passage carries its source URL.
3. **Answer** – `NCAIR1/N-ATLaS` writes a short answer in plain Nigerian English, citing sources, ending with a concrete next step.
4. **Safety** – refuses to collect PINs, OTPs, passwords or full BVN/NIN; redacts 10–11 digit numbers and phone numbers before anything is logged; says "I don't know" and routes to the agency when retrieval confidence is low.
5. **Agency dashboard** – anonymised topic trends, unanswered questions (knowledge gaps) and satisfaction, exportable as CSV. This is the evidence trail for the 50+ documented real-user interactions.

## Run locally

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...            # needs accepted access to the gated NCAIR models
python app.py
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` | `llamacpp` (CPU, GGUF quant of N-ATLaS) or `transformers` (GPU, full `NCAIR1/N-ATLaS`) |
| `GGUF_REPO` | `tosinamuda/N-ATLaS-GGUF` | GGUF repo for the CPU backend |
| `GGUF_FILE` | `*Q4_K_M.gguf` | GGUF file pattern |
| `LOG_DATASET_REPO` | unset | private HF dataset to persist anonymised interaction logs |

## Team

Akara Kpeale and Odunlade Tobiloba.

## Licence and attribution

Uses N-ATLaS and NigerianAccentedEnglish under the NCAIR/Awarri terms of use. Knowledge-base passages are summaries of public official documents; each cites its source.
