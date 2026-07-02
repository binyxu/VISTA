#!/usr/bin/env python3
"""Download GAIA attached files from the gated Hugging Face dataset."""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/gaia_2023_all_validation.jsonl")
    ap.add_argument("--dataset", default="gaia-benchmark/GAIA")
    ap.add_argument("--out-dir", default="data/attachments")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = ok = 0
    manifest = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            file_path = str(row.get("file_path") or "").strip()
            file_name = str(row.get("file_name") or "").strip()
            if not file_path:
                continue
            n += 1
            try:
                src = hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    filename=file_path,
                    token=token,
                )
                dest = out_dir / file_name
                shutil.copyfile(src, dest)
                ok += 1
                manifest.append({
                    "task_id": row.get("task_id"),
                    "file_name": file_name,
                    "file_path": file_path,
                    "local_path": str(dest),
                })
                print(f"OK {file_name}")
            except Exception as exc:
                print(f"FAIL {file_path}: {exc}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded {ok}/{n} attachments into {out_dir}")


if __name__ == "__main__":
    main()
