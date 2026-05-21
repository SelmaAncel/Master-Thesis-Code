import os
import json
import pickle
import numpy as np
from collections import defaultdict

SAVE_PATH = "D:/consistent_unseen_speakers.json"
MAX_UNSEEN_SPEAKERS = 100
SEED = 42

# Configurations for each clip lenght
DURATION_CONFIGS = {
    "0.5s": {
        "memmaps": ["D:/train_memmap_05", "D:/val_memmap_05", "D:/test_memmap_05"],
        "speaker_map": "D:/Embeddings05/pose_120/speaker_to_idx.json",
        "min_unseen_clips": 0,
        "max_unseen_clips": 0,
    },
    "1s": {
        "memmaps": ["D:/train_memmap_1", "D:/val_memmap_1", "D:/test_memmap_1"],
        "speaker_map": "D:/Embeddings1/pose_60/speaker_to_idx.json",
        "min_unseen_clips": 20,
        "max_unseen_clips": 20,
    },
    "2s": {
        "memmaps": ["D:/train_memmap_2", "D:/val_memmap_2", "D:/test_memmap_2"],
        "speaker_map": "D:/Embeddings2/pose_30/speaker_to_idx.json",
        "min_unseen_clips": 20,
        "max_unseen_clips": 20,
    },
    "4s": {
        "memmaps": ["D:/train_memmap_4", "D:/val_memmap_4", "D:/test_memmap_4"],
        "speaker_map": "D:/Embeddings4/pose_15/speaker_to_idx.json",
        "min_unseen_clips": 0,
        "max_unseen_clips": 0,
    },
    "8s": {
        "memmaps": ["D:/train_memmap_8", "D:/val_memmap_8", "D:/test_memmap_8"],
        "speaker_map": "D:/Embeddings8/pose_8/speaker_to_idx.json",
        "min_unseen_clips": 0,
        "max_unseen_clips": 0,
    },
    "10s": {
        "memmaps": ["D:/train_memmap_10", "D:/val_memmap_10", "D:/test_memmap_10"],
        "speaker_map": "D:/Embeddings10/pose_6/speaker_to_idx.json",
        "min_unseen_clips": 0,
        "max_unseen_clips": 0,
    },
}

# Count clips in memmap
def count_clips_in_memmaps(memmap_dirs):
    counts = defaultdict(int)
    for memmap_dir in memmap_dirs:
        with open(os.path.join(memmap_dir, "aux_info.pkl"), "rb") as f:
            aux_info = pickle.load(f)
        for entry in aux_info:
            counts[entry["vid"]] += 1
    return counts

# Main
def main():
    print("\n" + "="*60)
    print("  FINDING CONSISTENT UNSEEN SPEAKERS")
    print("="*60)

    per_duration = {}
    for dur, cfg in DURATION_CONFIGS.items():
        with open(cfg["speaker_map"]) as f:
            known = set(json.load(f).keys())
        counts = count_clips_in_memmaps(cfg["memmaps"])
        per_duration[dur] = {
            "known":         known,
            "counts":        counts,
            "min_unseen_clips": cfg["min_unseen_clips"],
            "max_unseen_clips": cfg["max_unseen_clips"],
        }
        print(f"  {dur:>4}  —  {len(known):>5} known speakers  |  "
              f"{len(counts):>6} total in memmap  |  "
              f"clips [{cfg['min_unseen_clips']}, {cfg['max_unseen_clips']}]")

    all_vids = set()
    for d in per_duration.values():
        all_vids |= set(d["counts"].keys())
    print(f"\n  Total unique speakers across all memmaps: {len(all_vids):,}")

    candidates = []
    for vid in sorted(all_vids):
        if not all(vid not in d["known"] for d in per_duration.values()):
            continue
        if not all(d["min_unseen_clips"] <= d["counts"].get(vid, 0)
                   for d in per_duration.values()):
            continue
        candidates.append(vid)

    print(f"  Candidates (UNSEEN everywhere + enough clips): {len(candidates)}")

    if len(candidates) == 0:
        raise RuntimeError(
            "No speakers qualify. Consider lowering min_unseen_clips values.")

    if len(candidates) < MAX_UNSEEN_SPEAKERS:
        print(f"  WARNING: Only {len(candidates)} candidates, "
              f"fewer than requested {MAX_UNSEEN_SPEAKERS}. Using all.")

    rng      = np.random.default_rng(SEED)
    n        = min(MAX_UNSEEN_SPEAKERS, len(candidates))
    selected = sorted(rng.choice(candidates, size=n, replace=False).tolist())
    print(f"  Selected {len(selected)} consistent UNSEEN speakers")

    # Show clip counts per duration for the selected speakers
    print(f"\n  Clip counts per duration (first 15 shown):")
    header = f"  {'Speaker':<24}" + "".join(f"  {d:>5}" for d in DURATION_CONFIGS)
    print(header)
    print("  " + "-" * (24 + 7 * len(DURATION_CONFIGS)))
    for vid in selected[:15]:
        row = f"  {vid:<24}"
        for d in per_duration.values():
            row += f"  {d['counts'].get(vid, 0):>5}"
        print(row)
    if len(selected) > 15:
        print(f"  ... ({len(selected) - 15} more)")

    os.makedirs(os.path.dirname(os.path.abspath(SAVE_PATH)), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump({
            "speakers":   selected,
            "n_speakers": len(selected),
            "seed":       SEED,
            "duration_min_unseen_clips": {
                dur: cfg["min_unseen_clips"] for dur, cfg in DURATION_CONFIGS.items()
            },
        }, f, indent=2)
    print(f"\n  Saved → {SAVE_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()