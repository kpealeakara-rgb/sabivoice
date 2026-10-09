from pathlib import Path
import re
import tempfile
"""SabiVoice: voice-first citizen helpline on N-ATLaS."""
import os
import time
from collections import Counter

import gradio as gr
import pandas as pd

import asr
import llm
import rag
import store
import tts
from safety import redact, mentions_secret

# Retrieval thresholds, tuned on real tester questions (Oct 9): in-scope questions
# scored 0.66-0.83, off-topic ones (criminal law, national anthem) 0.56-0.61.
MIN_SCORE = 0.64      # below this, retrieval is not trusted for an answer
POINTER_SCORE = 0.62  # between this and MIN_SCORE, point to the agency instead

SYSTEM = (
    "You are SabiVoice, a public-service helpline assistant for Nigerians. "
    "Answer ONLY from the numbered official passages provided. "
    "Write in plain, friendly Nigerian English that a secondary-school leaver understands: "
    "3 to 6 short sentences, no jargon, no markdown headings. "
    "Cite the passages you used like [1] or [2], but never use the word 'passage' in your answer. "
    "Do not state anything stronger or more general than the passages say. "
    "Only give phone numbers, USSD codes, websites, emails or addresses that appear word for word "
    "in the passages. Never make one up. "
    "End with one concrete next step (where to go, which code to dial, which office or website). "
    "If the passages do not answer the question, say you are not sure and point to the agency listed. "
    "Never ask for or accept a PIN, OTP, password, CVV or full BVN/NIN. "
    "If someone describes a message asking for those, tell them clearly it is likely a scam."
)

SECRET_WARNING = (
    "Please never share your PIN, OTP, password, CVV or full BVN with anyone, "
    "including people who say they are from your bank, NIMC or the government. "
    "No genuine agency will ask for these by call, SMS or WhatsApp. "
)

OUT_OF_SCOPE = (
    "Sorry, I can only help with government and citizen services for now: "
    "tax, NIN and BVN, pensions, health insurance, consumer complaints, "
    "phone network problems, and checking if a message is a scam. "
    "Please ask me about one of those."
)

_PHONE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")
_URL = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.(?:gov\.ng|com\.ng|org\.ng|ng|com|org|net)(?:/\S*)?", re.I)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _drop_invented_contacts(reply: str, context: str) -> str:
    """Remove sentences that carry a phone number or website not found in the sources."""
    ctx_digits = _digits(context)
    ctx_low = context.lower()
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", reply):
        bad = any(_digits(p) and _digits(p) not in ctx_digits for p in _PHONE.findall(sent))
        for u in _URL.findall(sent):
            host = re.sub(r"^(?:https?://)?(?:www\.)?", "", u.lower()).split("/")[0]
            if host and host not in ctx_low:
                bad = True
        if not bad and sent.strip():
            kept.append(sent.strip())
    return " ".join(kept).strip()


def answer(question: str):
    hits = rag.search(question, k=4)
    good = [(s, c) for s, c in hits if s >= MIN_SCORE]
    if not good:
        top_score = hits[0][0] if hits else 0.0
        top = hits[0][1] if hits else None
        if top and top_score >= POINTER_SCORE:
            msg = ("I'm not fully sure about this one, so I don't want to guess. "
                   f"The best people to ask are {top.agency}: {top.contact}")
            return msg, [], top.topic, top_score, False
        return OUT_OF_SCOPE, [], "Out of scope", top_score, False

    context = "\n\n".join(
        f"[{i}] ({c.topic} | {c.agency}) {c.title}\n{c.text}\nSource: {c.source}\nContact: {c.contact}"
        for i, (_, c) in enumerate(good, 1)
    )
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Official passages:\n{context}\n\nCitizen's question: {question}"},
    ]
    reply = _drop_invented_contacts(llm.chat(msgs), context)
    if not reply:
        top = good[0][1]
        reply = ("I'm not fully sure about this one, so I don't want to guess. "
                 f"The best people to ask are {top.agency}: {top.contact}")
    if mentions_secret(question) and "PIN" not in reply:
        reply = SECRET_WARNING + reply
    return reply, good, good[0][1].topic, good[0][0], True


def handle(consent, audio, typed):
    if not consent:
        raise gr.Error("Please tick the consent box first.")
    t0 = time.time()
    channel = "voice" if audio else "text"
    question = asr.transcribe(audio) if audio else (typed or "").strip()
    if not question:
        raise gr.Error("I couldn't hear a question. Please record again or type it.")

    reply, used, topic, score, grounded = answer(question)
    sources = "\n".join(
        f"[{i}] [{c.title}]({c.source}) \u00b7 {c.agency}" for i, (_, c) in enumerate(used, 1)
    ) or "_No matching official source found._"
    audio_out = tts.speak(reply)

    iid = store.log_interaction(
        channel=channel,
        question=redact(question),
        answer=redact(reply),
        topic=topic,
        score=round(score, 3),
        grounded=grounded,
        sources=[c.source for _, c in used],
        latency_s=round(time.time() - t0, 1),
    )
    return question, reply, sources, audio_out, iid, gr.update(visible=True)


def feedback(iid, helpful, comment):
    if iid:
        store.log_feedback(iid, helpful, redact(comment or ""))
    return gr.update(value="Thank you, your feedback helps us improve."), gr.update(visible=False)


def dashboard():
    rows = store.interactions()
    fb = {f["id"]: f for f in store.feedback()}
    if not rows:
        empty = pd.DataFrame(columns=["topic", "questions"])
        return "No interactions yet.", empty, pd.DataFrame(), pd.DataFrame(), None
    df = pd.DataFrame(rows)
    df["helpful"] = df["id"].map(lambda i: fb.get(i, {}).get("helpful"))
    total = len(df)
    voice = int((df["channel"] == "voice").sum())
    rated = df["helpful"].dropna()
    sat = f"{100 * rated.mean():.0f}%" if len(rated) else "n/a"
    gaps = int((~df["grounded"]).sum())
    summary = (
        f"**{total}** interactions \u00b7 **{voice}** by voice \u00b7 satisfaction **{sat}** "
        f"({len(rated)} rated) \u00b7 **{gaps}** knowledge gaps \u00b7 median latency "
        f"**{df['latency_s'].median():.1f}s**"
    )
    by_topic = (pd.DataFrame(Counter(df["topic"]).most_common(), columns=["topic", "questions"]))
    gap_df = df[~df["grounded"]][["ts", "topic", "question"]].tail(25)
    recent = df[["ts", "channel", "topic", "question", "helpful"]].tail(25).iloc[::-1]
    path = Path(tempfile.gettempdir()) / "sabivoice_interactions_export.csv"
    df.to_csv(path, index=False)
    return summary, by_topic, gap_df, recent, str(path)


THEME = gr.themes.Soft(primary_hue="green", secondary_hue="emerald", neutral_hue="slate")

with gr.Blocks(theme=THEME, title="SabiVoice") as demo:
    gr.Markdown(
        "# \U0001f399\ufe0f SabiVoice\n"
        "**Ask about tax, NIN, BVN, health insurance, pensions, your rights as a customer, "
        "or a message you think is a scam. Speak in your normal Nigerian English.**"
    )
    iid = gr.State("")
    with gr.Tab("Ask"):
        consent = gr.Checkbox(
            label="I agree that my question (with phone and ID numbers removed) may be stored "
                  "anonymously to improve this service. I will not share PINs, OTPs or passwords.",
        )
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                 label="Record your question")
                typed = gr.Textbox(label="\u2026or type it", lines=2,
                                   placeholder="e.g. Do I still pay tax if I earn 70k a month?")
                ask = gr.Button("Ask SabiVoice", variant="primary")
                gr.Examples(
                    examples=[
                        [None, "Someone called me say my BVN don block and I should send OTP. Wetin I go do?"],
                        [None, "How do I check my NIN on my phone?"],
                        [None, "My employer deducts pension but I can't see it. Who do I report to?"],
                    ],
                    inputs=[audio, typed],
                )
            with gr.Column():
                heard = gr.Textbox(label="What I heard", interactive=False)
                reply = gr.Textbox(label="Answer", lines=7, interactive=False)
                spoken = gr.Audio(label="Listen", autoplay=True, interactive=False)
                srcs = gr.Markdown()
                with gr.Group(visible=False) as fb_box:
                    gr.Markdown("**Was this helpful?**")
                    comment = gr.Textbox(label="Anything we should fix? (optional)")
                    with gr.Row():
                        yes = gr.Button("\U0001f44d Yes")
                        no = gr.Button("\U0001f44e No")
                thanks = gr.Markdown()
        ask.click(handle, [consent, audio, typed], [heard, reply, srcs, spoken, iid, fb_box])
        yes.click(lambda i, c: feedback(i, True, c), [iid, comment], [thanks, fb_box])
        no.click(lambda i, c: feedback(i, False, c), [iid, comment], [thanks, fb_box])

    with gr.Tab("Agency dashboard"):
        gr.Markdown("Anonymised view of what citizens are asking, where answers are missing, "
                    "and how useful the answers were.")
        refresh = gr.Button("Refresh")
        summary = gr.Markdown()
        with gr.Row():
            topic_tbl = gr.Dataframe(label="Questions by topic")
            gaps_tbl = gr.Dataframe(label="Knowledge gaps (no official answer found)")
        recent_tbl = gr.Dataframe(label="Recent questions")
        export = gr.File(label="Export all interactions (CSV)")
        refresh.click(dashboard, None, [summary, topic_tbl, gaps_tbl, recent_tbl, export])
        demo.load(dashboard, None, [summary, topic_tbl, gaps_tbl, recent_tbl, export])

    with gr.Tab("About"):
        gr.Markdown(open("README.md", encoding="utf-8").read().split("---", 2)[-1])

if __name__ == "__main__":
    demo.queue(max_size=32).launch(share=os.getenv("SHARE") == "1", show_error=True)
