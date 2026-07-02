#!/usr/bin/env python3
"""Download gated GAIA splits to local JSONL.

Requires a Hugging Face token with access to gaia-benchmark/GAIA. The token is
read from HF_TOKEN/HUGGINGFACE_HUB_TOKEN or the local huggingface-cli login.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gaia-benchmark/GAIA")
    parser.add_argument("--config", default="2023_all")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or True
    ds = load_dataset(args.dataset, args.config, split=args.split, token=token)

    out = Path(args.out or f"data/gaia_{args.config}_{args.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    print(f"WROTE {out} n={len(ds)}")


if __name__ == "__main__":
    main()
