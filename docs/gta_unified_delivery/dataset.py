"""Portable entry point: one root for every task and release."""
from pathlib import Path
from dataset_core import MixedMemoryDataset


def load_dataset(root, manifest='train_round_00', temporal_indices=None):
    root = Path(root).resolve()
    return MixedMemoryDataset(root/'manifests'/(manifest+'.jsonl'), {'data':root}, temporal_indices)
