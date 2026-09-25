"""GDM65 entrypoint reusing the canonical GDM64 numerical wrapper.

The wrapper delegates through followup64.execute without rebinding the real GDM64
run module's main function. This keeps GDM65's own delegation into the original
GDM64 runner non-recursive while preserving the canonical numerical wrapper.
"""
from types import SimpleNamespace

from experiments.followup64 import execute as e64
from experiments.followup65 import run as r65


def main():
    old_runner = e64.g64
    e64.g64 = SimpleNamespace(main=r65.main)
    try:
        e64.main()
    finally:
        e64.g64 = old_runner


if __name__ == "__main__":
    main()
