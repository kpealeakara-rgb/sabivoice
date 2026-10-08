"""Retrieval over the curated official-source knowledge base in kb/.

Each kb/*.md file starts with front matter:
---
topic: Tax
agency: Nigeria Revenue Service (NRS)
contact: https://...
---
followed by '## ' sections. Every section must end with a line 'Source: <url>'.
"""
import os
import re
import glob
import functools
from dataclasses import dataclass

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
KB_DIR = os.path.join(os.path.dirname(__file__), "kb")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    topic: str
    agency: str
    contact: str
    title: str
    text: str
    source: str


def _parse(path: str) -> list[Chunk]:
    raw = open(path, encoding="utf-8").read()
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        raw = raw[m.end():]
    chunks = []
    for sec in re.split(r"\n(?=## )", raw):
        sec = sec.strip()
        if not sec.startswith("## "):
            continue
        title, _, body = sec[3:].partition("\n")
        src = re.search(r"^Source:\s*(\S+)", body, re.M)
        body = re.sub(r"^Source:.*$", "", body, flags=re.M).strip()
        chunks.append(Chunk(
            topic=meta.get("topic", "General"),
            agency=meta.get("agency", ""),
            contact=meta.get("contact", ""),
            title=title.strip(),
            text=body,
            source=src.group(1) if src else meta.get("contact", ""),
        ))
    return chunks


@functools.lru_cache(maxsize=1)
def _index():
    from sentence_transformers import SentenceTransformer

    chunks = []
    for p in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        if os.path.basename(p).lower() == "readme.md":
            continue
        chunks.extend(_parse(p))
    model = SentenceTransformer(EMBED_MODEL)
    docs = [f"{c.topic}. {c.title}. {c.text}" for c in chunks]
    emb = model.encode(docs, normalize_embeddings=True) if docs else np.zeros((0, 384))
    return model, chunks, np.asarray(emb)


def search(query: str, k: int = 4) -> list[tuple[float, Chunk]]:
    model, chunks, emb = _index()
    if not chunks:
        return []
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    scores = emb @ q
    order = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in order]


def topics() -> list[str]:
    return sorted({c.topic for c in _index()[1]})
