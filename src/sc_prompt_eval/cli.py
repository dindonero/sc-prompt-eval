from __future__ import annotations

import argparse
from .runner.run import main

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run smart contract vulnerability detection experiments with LLM prompts"
    )
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument("--dry-run", action="store_true", help="Validate config and prompt rendering only")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose output")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for contract processing (default: 1 for sequential)"
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config, dry_run=args.dry_run, verbose=args.verbose, workers=args.workers)
