#!/usr/bin/env python3
"""Compact a LOCA/GEM output directory with debug-preserving artifacts.

Default policy:
  - Preserve trajectories and per-step API inputs (`logs/context_step_*.json`)
    as gzip-compressed files.
  - Preserve result/config/eval/token files, agent workspace, and context
    workspace as gzip-compressed files.
  - Replace large reconstructable input/state directories (`local_db`, `files`,
    `initial_workspace`, `groundtruth_workspace`, `preprocess`) with manifest
    entries containing hashes and sizes.

The compact output is intentionally not a byte-for-byte self-contained backup
unless `--mode full-lossless` is used. It is a debug-preserving archive: it keeps
the agent trajectory and model inputs while treating benchmark inputs/databases
as rebuildable from code/config/data versions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


Decision = Literal["archive_gzip", "manifest_only", "skip"]


DEFAULT_MANIFEST_ONLY_DIRS = {
    "local_db",
    "files",
    "initial_workspace",
    "groundtruth_workspace",
    "preprocess",
    ".pdf_tools_tempfiles",
}

DEFAULT_ARCHIVE_DIRS = {
    "agent_workspace",
    "context_workspace",
    "logs",
    "evaluation",
}

DEFAULT_ARCHIVE_FILENAMES = {
    "results.json",
    "config_self_managed.json",
    "task_mapping.json",
    "trajectory.json",
    "eval.json",
    "token_stats.json",
    "task_config_generated.json",
    "emails_config.json",
    "email_config.json",
}

DEBUG_LOG_PREFIX = "context_step_"


@dataclass
class Counters:
    total_files: int = 0
    total_bytes: int = 0
    archived_files: int = 0
    archived_raw_bytes: int = 0
    archived_bytes: int = 0
    manifest_only_files: int = 0
    manifest_only_bytes: int = 0
    skipped_files: int = 0
    skipped_bytes: int = 0
    by_decision: dict[str, int] = field(default_factory=dict)

    def add_source(self, size: int) -> None:
        self.total_files += 1
        self.total_bytes += size

    def add_archive(self, raw_size: int, stored_size: int) -> None:
        self.archived_files += 1
        self.archived_raw_bytes += raw_size
        self.archived_bytes += stored_size
        self.by_decision["archive_gzip"] = self.by_decision.get("archive_gzip", 0) + raw_size

    def add_manifest_only(self, size: int) -> None:
        self.manifest_only_files += 1
        self.manifest_only_bytes += size
        self.by_decision["manifest_only"] = self.by_decision.get("manifest_only", 0) + size

    def add_skip(self, size: int) -> None:
        self.skipped_files += 1
        self.skipped_bytes += size
        self.by_decision["skip"] = self.by_decision.get("skip", 0) + size


def rel_parts(path: Path, root: Path) -> tuple[str, ...]:
    return path.relative_to(root).parts


def is_under_any(parts: tuple[str, ...], names: set[str]) -> bool:
    return any(part in names for part in parts)


def should_archive_debug_input(path: Path) -> bool:
    return path.name.startswith(DEBUG_LOG_PREFIX) and path.suffix == ".json"


def decide_file(
    path: Path,
    root: Path,
    mode: str,
    archive_dirs: set[str],
    manifest_only_dirs: set[str],
) -> tuple[Decision, str]:
    parts = rel_parts(path, root)
    name = path.name

    if mode == "full-lossless":
        return "archive_gzip", "full_lossless"

    if name in DEFAULT_ARCHIVE_FILENAMES:
        return "archive_gzip", f"important_file:{name}"

    if should_archive_debug_input(path):
        return "archive_gzip", "api_input_context_step"

    if is_under_any(parts, archive_dirs):
        return "archive_gzip", "debug_or_workspace_dir"

    if is_under_any(parts, manifest_only_dirs):
        return "manifest_only", "reconstructable_dir"

    if mode == "debug-reconstructable":
        # Unknown files are safer to keep. This avoids silently dropping new
        # debug artifacts introduced by future runners.
        return "archive_gzip", "unknown_keep_safe"

    if mode == "minimal-reconstructable":
        return "manifest_only", "minimal_unknown_manifest"

    raise ValueError(f"unknown mode: {mode}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_copy(src: Path, dst: Path, compresslevel: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, gzip.open(dst, "wb", compresslevel=compresslevel) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    return dst.stat().st_size


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            yield path


def manifest_entry(
    src: Path,
    root: Path,
    decision: Decision,
    reason: str,
    sha256: str,
    archived_relpath: str | None,
) -> dict:
    stat = src.stat()
    return {
        "path": str(src.relative_to(root)),
        "decision": decision,
        "reason": reason,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode,
        "sha256": sha256,
        "archived_path": archived_relpath,
        "restore": (
            {"type": "archive_gzip", "path": archived_relpath}
            if decision == "archive_gzip"
            else {
                "type": "external_reconstruct",
                "require_sha256": sha256,
                "note": "Rebuild or copy this file from the original benchmark data/preprocess output, then verify sha256.",
            }
        ),
    }


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def compact(args: argparse.Namespace) -> int:
    src_root = args.input.resolve()
    out_root = args.output.resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"input is not a directory: {src_root}")
    if out_root == src_root or src_root in out_root.parents:
        raise SystemExit("output must not be the input directory or inside it")
    if out_root.exists() and any(out_root.iterdir()) and not args.force and not args.dry_run:
        raise SystemExit(f"output exists and is not empty: {out_root} (use --force)")

    archive_dirs = set(DEFAULT_ARCHIVE_DIRS)
    archive_dirs.update(args.archive_dir or [])
    manifest_only_dirs = set(DEFAULT_MANIFEST_ONLY_DIRS)
    manifest_only_dirs.update(args.manifest_only_dir or [])
    manifest_only_dirs.difference_update(archive_dirs)

    counters = Counters()
    start = time.time()
    manifest_rows: list[dict] = []
    manifest_path = out_root / "manifest.jsonl"
    blobs_root = out_root / "files"

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        if args.force:
            for child in out_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    for src in iter_files(src_root):
        size = src.stat().st_size
        counters.add_source(size)
        decision, reason = decide_file(
            src,
            src_root,
            mode=args.mode,
            archive_dirs=archive_dirs,
            manifest_only_dirs=manifest_only_dirs,
        )
        digest = sha256_file(src)
        archived_relpath = None

        if decision == "archive_gzip":
            rel = src.relative_to(src_root)
            dst = blobs_root / rel
            dst = dst.with_name(dst.name + ".gz")
            archived_relpath = str(dst.relative_to(out_root))
            stored_size = 0
            if not args.dry_run:
                stored_size = gzip_copy(src, dst, args.compresslevel)
            counters.add_archive(size, stored_size)
        elif decision == "manifest_only":
            counters.add_manifest_only(size)
        else:
            counters.add_skip(size)

        row = manifest_entry(src, src_root, decision, reason, digest, archived_relpath)
        manifest_rows.append(row)

        if args.progress and counters.total_files % args.progress == 0:
            print(
                f"processed {counters.total_files} files; "
                f"source={counters.total_bytes / 1024 / 1024:.1f} MiB; "
                f"manifest_only={counters.manifest_only_bytes / 1024 / 1024:.1f} MiB",
                file=sys.stderr,
            )

    elapsed = time.time() - start
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "input": str(src_root),
        "output": str(out_root),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 3),
        "gzip_compresslevel": args.compresslevel,
        "archive_dirs": sorted(archive_dirs),
        "manifest_only_dirs": sorted(manifest_only_dirs),
        "total_files": counters.total_files,
        "total_bytes": counters.total_bytes,
        "archived_files": counters.archived_files,
        "archived_raw_bytes": counters.archived_raw_bytes,
        "archived_bytes": counters.archived_bytes,
        "manifest_only_files": counters.manifest_only_files,
        "manifest_only_bytes": counters.manifest_only_bytes,
        "skipped_files": counters.skipped_files,
        "skipped_bytes": counters.skipped_bytes,
        "estimated_output_bytes": None if args.dry_run else 0,
        "information_policy": {
            "trajectory": "preserved as gzip",
            "api_inputs": "logs/context_step_*.json preserved as gzip",
            "agent_workspace": "preserved as gzip by default",
            "context_workspace": "preserved as gzip by default",
            "local_db_files_initial_groundtruth": "manifest/hash only by default; reconstruct externally and verify sha256",
        },
    }

    if args.dry_run:
        summary["estimated_output_bytes"] = None
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        return 0

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    write_json(out_root / "summary.json", summary)
    write_restore_script(out_root / "restore_archived_files.py")
    summary["estimated_output_bytes"] = sum(p.stat().st_size for p in out_root.rglob("*") if p.is_file())
    write_json(out_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def write_restore_script(path: Path) -> None:
    script = r'''#!/usr/bin/env python3
"""Restore gzip-archived files from a compact_loca_output archive.

Manifest-only files are not restored by this script. Rebuild/copy them from the
benchmark source, then verify their sha256 against manifest.jsonl.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    manifest = args.archive / "manifest.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    restored = 0
    manifest_only = 0
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["decision"] != "archive_gzip":
                manifest_only += 1
                continue
            src = args.archive / row["archived_path"]
            dst = args.output / row["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
            dst.chmod(row["mode"] & 0o777)
            mtime = row["mtime_ns"] / 1_000_000_000
            os.utime(dst, (mtime, mtime))
            if args.verify and sha256_file(dst) != row["sha256"]:
                raise SystemExit(f"sha256 mismatch: {dst}")
            restored += 1
    print(f"restored archived files: {restored}; manifest-only files not restored: {manifest_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact a LOCA/GEM output directory while preserving trajectory and API input debug artifacts.",
    )
    parser.add_argument("input", type=Path, help="Result directory to compact")
    parser.add_argument("output", type=Path, help="New compact archive directory")
    parser.add_argument(
        "--mode",
        choices=["debug-reconstructable", "minimal-reconstructable", "full-lossless"],
        default="debug-reconstructable",
        help="Compaction policy. Default preserves trajectory/API-input debug data and manifests rebuildable inputs.",
    )
    parser.add_argument(
        "--archive-dir",
        action="append",
        help="Additional directory basename to gzip-archive instead of manifest-only. Can be repeated.",
    )
    parser.add_argument(
        "--manifest-only-dir",
        action="append",
        help="Additional directory basename to replace with manifest/hash only. Can be repeated.",
    )
    parser.add_argument("--compresslevel", type=int, default=6, choices=range(1, 10))
    parser.add_argument("--dry-run", action="store_true", help="Scan and print summary without writing output")
    parser.add_argument("--force", action="store_true", help="Replace existing output contents")
    parser.add_argument("--progress", type=int, default=1000, help="Print progress every N files; 0 disables")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(compact(parse_args(sys.argv[1:])))
