"""
consolidate_sweep.py

Converts raw sweep30 JSONL files into canonical condition files.

Input:  data/sweep30_cond{1-4}.jsonl
Output: data/sweep_cond_{A-D}.jsonl  (deduplicated, with metadata fields)
Archive: data/sweep_archive/          (copies of originals)

Canonical mapping (from PIPELINE.md):
  cond_A <- sweep30_cond4  beta=0.10  kl_coeff=1.0  n_epochs=3
  cond_B <- sweep30_cond1  beta=0.10  kl_coeff=2.0  n_epochs=3
  cond_C <- sweep30_cond2  beta=0.05  kl_coeff=3.0  n_epochs=5
  cond_D <- sweep30_cond3  beta=0.02  kl_coeff=3.0  n_epochs=8

Usage:
  python consolidate_sweep.py [--data-dir data] [--dry-run]
"""

import argparse
import json
import shutil
from pathlib import Path

CANONICAL = {
    "A": {"source": "sweep30_cond4", "beta": 0.10, "kl_coeff": 1.0, "n_epochs": 3},
    "B": {"source": "sweep30_cond1", "beta": 0.10, "kl_coeff": 2.0, "n_epochs": 3},
    "C": {"source": "sweep30_cond2", "beta": 0.05, "kl_coeff": 3.0, "n_epochs": 5},
    "D": {"source": "sweep30_cond3", "beta": 0.02, "kl_coeff": 3.0, "n_epochs": 8},
}


def consolidate(data_dir: Path, dry_run: bool) -> None:
    archive_dir = data_dir / "sweep_archive"
    if not dry_run:
        archive_dir.mkdir(exist_ok=True)

    for label, meta in CANONICAL.items():
        source_file = data_dir / f"{meta['source']}.jsonl"
        out_file = data_dir / f"sweep_cond_{label}.jsonl"

        if not source_file.exists():
            print(f"[WARN] {source_file} not found — skipping cond_{label}")
            continue

        seen = set()
        records = []
        with source_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (record.get("id"), record.get("step_idx"))
                if key in seen:
                    continue
                seen.add(key)
                record["sweep_condition"] = f"cond_{label}"
                record["beta"] = meta["beta"]
                record["kl_coeff"] = meta["kl_coeff"]
                record["n_epochs"] = meta["n_epochs"]
                records.append(record)

        print(f"cond_{label}: {len(records)} records from {source_file.name} -> {out_file.name}")
        if not dry_run:
            with out_file.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            shutil.copy2(source_file, archive_dir / source_file.name)
            print(f"  archived {source_file.name} -> {archive_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="Directory containing sweep JSONL files (default: data)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: data directory not found: {data_dir}")

    if args.dry_run:
        print("[dry-run] no files will be written\n")

    consolidate(data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
