from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .oracle import OracleTrackingPipeline, aggregate_oracle_metrics
from .pipeline import BaselinePipeline
from .submission import audit_submission


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="JSON configuration; defaults to built-in values")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biohub-track")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the baseline over competition test Zarr stores")
    run.add_argument("--test-dir", required=True)
    run.add_argument("--output", default="submission.csv")
    _add_config_argument(run)
    run.add_argument("--stats", default="run_stats.csv")

    oracle = sub.add_parser(
        "oracle-eval",
        help="evaluate the geometric tracker using ground-truth GEFF centroids",
    )
    oracle.add_argument("--train-dir", required=True)
    oracle.add_argument("--output", default="oracle_metrics.csv")
    oracle.add_argument("--predictions-dir")
    _add_config_argument(oracle)

    audit = sub.add_parser("audit", help="audit an existing submission.csv")
    audit.add_argument("path")
    return parser


def _load_config(path: str | None) -> PipelineConfig:
    return PipelineConfig.from_json(path) if path else PipelineConfig()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        submission, stats = BaselinePipeline(_load_config(args.config)).run(
            args.test_dir,
            args.output,
        )
        stats.to_csv(args.stats, index=False)
        print(f"Wrote {args.output}: {len(submission)} rows")
        print(stats.to_string(index=False))
        return

    if args.command == "oracle-eval":
        metrics = OracleTrackingPipeline(_load_config(args.config)).run(
            args.train_dir,
            output_path=args.output,
            predictions_dir=args.predictions_dir,
        )
        print(metrics.to_string(index=False))
        print(json.dumps(aggregate_oracle_metrics(metrics), indent=2))
        return

    frame = pd.read_csv(Path(args.path))
    report = audit_submission(frame)
    print(json.dumps(report.to_dict(), indent=2))
    raise SystemExit(0 if report.valid else 2)


if __name__ == "__main__":
    main()
