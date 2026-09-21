"""Seed-stable GDM45 CLI wrapper.

`run.run_one` constructs the learned bundle before entering its training function, so
this wrapper seeds all RNGs immediately before every arm/seed construction. The
wrapper is the workflow entrypoint; direct `run.py` execution is not used by CI.
"""
import random
import numpy as np
import torch

from . import run

_original_run_one = run.run_one


def _seeded_run_one(arm_name, task, seed, *args, **kwargs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return _original_run_one(arm_name, task, seed, *args, **kwargs)


def main():
    run.run_one = _seeded_run_one
    run.main()


if __name__ == "__main__":
    main()
