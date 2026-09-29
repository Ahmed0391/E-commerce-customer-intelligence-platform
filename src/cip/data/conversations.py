"""Fill conversation text.

The simulator decides WHEN a customer contacts support and ABOUT WHAT (intent).
This module decides WHAT THEY WROTE.

Primary source: the Bitext customer-support dataset (real, human-curated utterances
labelled with intent). Using real text instead of LLM-generated text avoids a
circular evaluation where the same kind of model writes and reads the data.

Known limitation: the sampled utterance is chosen by intent only, so its tone is NOT
controlled by the latent 'frustration' value. That is why sentiment is evaluated on
a hand-labelled sample, not on generator labels.

Fallback source: tiny templates, for offline smoke tests ONLY. Never report metrics on them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BITEXT_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")

_TEMPLATES = {
    "track_order": ["where is my order {order}?", "can you tell me where order {order} is"],
    "delivery_period": ["how long until order {order} arrives?", "when will my package be delivered"],
    "complaint": ["my order {order} arrived damaged, this is unacceptable", "I want to file a complaint about order {order}"],
    "get_refund": ["I want a refund for order {order}", "how do I get my money back for {order}"],
    "delete_account": ["please close my account", "I want to delete my account"],
}
_GENERIC = "I have a question about {intent}"


def _fill_placeholders(text: str, order_id) -> str:
    def repl(m):
        key = m.group(1).lower()
        if "order" in key and pd.notna(order_id):
            return f"#{int(order_id)}"
        return m.group(1).lower()   # e.g. {{Account Type}} -> "account type"
    return _PLACEHOLDER.sub(repl, text)


def load_bitext() -> pd.DataFrame:
    from datasets import load_dataset  # optional dependency: pip install -e ".[text]"
    df = load_dataset(BITEXT_ID, split="train").to_pandas()
    return df[["instruction", "intent"]]


def fill_text(convs: pd.DataFrame, labels: pd.DataFrame, source: str = "bitext", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    merged = convs.drop(columns="text").merge(labels[["conversation_id", "intent"]], on="conversation_id")
    texts, sources = [], []

    if source == "bitext":
        corpus = load_bitext()
        pools = {k: g["instruction"].to_numpy() for k, g in corpus.groupby("intent")}
        missing = set(merged["intent"]) - set(pools)
        if missing:
            raise ValueError(f"Intents not in Bitext: {sorted(missing)}. Fix configs/generator.yaml.")
        for r in merged.itertuples(index=False):
            texts.append(_fill_placeholders(rng.choice(pools[r.intent]), r.order_id))
            sources.append("bitext")
    elif source == "templates":
        for r in merged.itertuples(index=False):
            options = _TEMPLATES.get(r.intent, [_GENERIC])
            order = f"#{int(r.order_id)}" if pd.notna(r.order_id) else "my order"
            texts.append(rng.choice(options).format(order=order, intent=r.intent.replace("_", " ")))
            sources.append("templates")
    else:
        raise ValueError(f"unknown source {source!r}")

    out = convs.copy()
    order = merged.set_index("conversation_id").index
    out = out.set_index("conversation_id").loc[order].reset_index()
    out["text"] = texts
    out["text_source"] = sources
    return out
