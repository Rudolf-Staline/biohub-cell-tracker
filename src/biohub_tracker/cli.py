from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .config import PipelineConfig
from .pipeline import BaselinePipeline
from .submission import audit_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biohub-track")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the baseline over competition test Zarr stores")
    run.add_argument("--test-dir", required=True)
    run.add_argument("--output", default="submission.csv")
    run.add_argument("--config")
    run.add_argument("--stats", default="run_stats.csv")

    audit = sub.add_parser("audit", help="audit an existing submission.csv")
    audit.add_argument("path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
        submission, stats = BaselinePipeline(config).run(args.test_dir, args.output)
        stats.to_csv(args.stats, index=False)
        print(f"Wrote {args.output}: {len(submission)} rows")
        print(stats.to_string(index=False))
        return
    frame = pd.read_csv(Path(args.path))
    report = audit_submission(frame)
    print(json.dumps(report.to_dict(), indent=2))
    raise SystemExit(0 if report.valid else 2)


if __name__ == "__main__":
    main()
