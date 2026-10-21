"""Run a local training configuration through the public Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

from trackmaniarl import RunSpec, Trainer, resolve_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", type=Path, help="Full training checkpoint")
    args = parser.parse_args()
    config = args.config.resolve()
    run = resolve_run(RunSpec.from_yaml(config), base_dir=config.parent)
    try:
        result = Trainer(run, resume_checkpoint=args.resume).train()
        print(result)
    finally:
        run.logger.close()


if __name__ == "__main__":
    main()
