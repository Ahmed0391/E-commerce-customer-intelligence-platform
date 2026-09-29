"""Generate the synthetic dataset.

    python scripts/generate_data.py                       # Bitext text (needs internet once)
    python scripts/generate_data.py --text-source templates   # offline smoke test only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cip.data.conversations import fill_text
from cip.data.generator import generate, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generator.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--text-source", choices=["bitext", "templates"], default="bitext")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tables = generate(cfg, args.out)
    convs = fill_text(tables["raw/conversations"], tables["ground_truth/conversation_labels"],
                      source=args.text_source, seed=cfg["seed"])
    convs.to_parquet(Path(args.out) / "raw/conversations.parquet", index=False)

    for name, df in tables.items():
        print(f"{name:38s} {len(df):>8,d} rows")
    print(f"\nconversation text source: {args.text_source}")


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
