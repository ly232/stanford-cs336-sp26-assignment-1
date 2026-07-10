#!/usr/bin/env python3
"""Regenerate the test_train_bpe_special_tokens snapshot.

Run this when the BPE training logic changes and the existing
snapshot no longer matches the current output.

Usage:
    uv run python cs336_basics/tokenizer/regenerate_snapshot.py
"""

import pickle
from pathlib import Path

from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH


def main():
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    data = {
        "vocab_keys": set(vocab.keys()),
        "vocab_values": set(vocab.values()),
        "merges": merges,
    }

    snapshot_path = (
        FIXTURES_PATH.parent / "_snapshots" / "test_train_bpe_special_tokens.pkl"
    )
    with open(snapshot_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Snapshot updated: {snapshot_path}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Merges count: {len(merges)}")


if __name__ == "__main__":
    main()
