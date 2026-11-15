"""Single CLI entry point: python -m eval."""
from __future__ import annotations

import argparse
from pathlib import Path

from eval import runner


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eval")
    parser.add_argument("--fixtures", action="store_true")
    parser.add_argument("--context", choices=("dev", "final"), default="dev")
    parser.add_argument("--out", default="eval/reports/latest.json")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    if args.fixtures:
        report, results = runner.run_fixtures(Path(args.out))
        print(f"wrote {args.out}: {len(results)} fixtures")
    else:
        report, results = runner.run_context(
            args.context, Path(args.out), diagnostic=args.diagnostic
        )
        print(f"wrote {args.out}: {len(results)} items ({args.context})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
