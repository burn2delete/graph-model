"""Canonical GitHub-Actions entrypoint for GDM47.

GDM47's semantic curriculum is keyed internally by tuple path suffixes because the
training generator consumes tuple paths. The shared evidence hash helper intentionally
accepts ordinary JSON values only. This entrypoint canonicalizes tuple-keyed mappings
for hashing without changing the shared GDM41–46 hashing behavior or dataset hashes.
"""
from __future__ import annotations

from experiments.followup47 import run

_base_sha = run.sha


def _json_hashable(value):
    if isinstance(value, dict):
        return {
            ("path:" + ".".join(key) if isinstance(key, tuple) else key): _json_hashable(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [*_json_hashable(list(value))]
    if isinstance(value, list):
        return [_json_hashable(item) for item in value]
    return value


def sha(value):
    return _base_sha(_json_hashable(value))


def main():
    # Scope the compatibility fix to GDM47 only. Existing measured experiment hashes
    # are deliberately not redefined.
    run.sha = sha
    run.main()


if __name__ == "__main__":
    main()
