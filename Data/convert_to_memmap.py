"""One-time conversion: source LMDB → memory-mapped numpy arrays.

Run this once in the Python 3.7 environment (which has pyarrow 0.15.0),
then use MemmapSpeechMotionDataset for training in any Python version.

Usage:
    python scripts/convert_to_memmap.py --config=config/pose_diffusion_expressive.yml
"""
import os
import sys
import pickle

import lmdb
import numpy as np
from tqdm import tqdm

[sys.path.append(i) for i in ['.', '..']]

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

print("convert_to_memmap")

import scripts.utils.data_utils_expressive as data_utils_expressive
from scripts.data_loader.data_preprocessor_expressive import DataPreprocessor
from scripts.parse_args_diffusion import parse_args


# ---------------------------------------------------------------------------
# Helpers: auto-detect key format and serialization used by the LMDB cache
# ---------------------------------------------------------------------------

def _try_deserialize(raw_bytes):
    """Try pyarrow first (used by lmdb_data_loader_expressive), then pickle."""
    # pyarrow (most common in HA2G / Gesture Generation codebases)
    try:
        import pyarrow as pa
        return pa.deserialize(raw_bytes)
    except Exception:
        pass
    # plain pickle fallback
    try:
        return pickle.loads(raw_bytes)
    except Exception:
        pass
    raise RuntimeError(
        "Could not deserialize LMDB value with either pyarrow or pickle.\n"
        "Run inspect_cache.py to investigate the raw format."
    )


def _make_key(idx, fmt):
    """Return the key bytes for index `idx` given the detected format string."""
    return fmt.format(idx).encode('ascii')


def _detect_key_format(txn, n_samples):
    """
    Probe the LMDB to figure out which zero-padded key format was used.
    Checks widths 4, 6, 8, 10 and also plain integer strings.
    Returns a Python format string like '{:010}' or '{:06}'.
    """
    cursor = txn.cursor()
    cursor.first()
    first_key = cursor.key().decode('ascii', errors='replace')
    print(f"  First raw key: {first_key!r}")

    candidates = ['{:010}', '{:08}', '{:06}', '{:04}', '{}']
    for fmt in candidates:
        key0 = fmt.format(0).encode('ascii')
        key1 = fmt.format(1).encode('ascii')
        if txn.get(key0) is not None and (n_samples < 2 or txn.get(key1) is not None):
            print(f"  Detected key format: {fmt!r}")
            return fmt

    # Last resort: use whatever the cursor gave us as a template
    width = len(first_key)
    fmt = '{:0' + str(width) + '}'
    print(f"  Falling back to width-{width} zero-padded format: {fmt!r}")
    return fmt


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(args):
    lmdb_dir = args.train_data_path[0]
    cache_dir = "D:/val_cache"
    os.makedirs(cache_dir, exist_ok=True)

    out_dir = "D:/val_memmap_10"
    os.makedirs(out_dir, exist_ok=True)

    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    n_poses = args.n_poses
    pose_resampling_fps = args.motion_resampling_framerate

    expected_audio_length = int(
        round(n_poses / pose_resampling_fps * 16000))
    expected_spectrogram_length = (
        data_utils_expressive.calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps))

    # Build cache if it doesn't exist (requires pyarrow 0.15.0)
    if not os.path.exists(cache_dir):
        print(f'Cache not found at {cache_dir}, building it...')
        if mean_dir_vec.shape[-1] != 3:
            mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
        n_poses_extended = int(round(n_poses * 1.25))
        data_sampler = DataPreprocessor(
            lmdb_dir, cache_dir, n_poses_extended,
            args.subdivision_stride, pose_resampling_fps,
            args.mean_pose, mean_dir_vec)
        data_sampler.run()

    # Open cache LMDB
    env = lmdb.open(cache_dir, readonly=True, lock=False)
    with env.begin() as txn:
        n_samples = txn.stat()['entries']
    print(f'Found {n_samples} samples in cache')

    # -----------------------------------------------------------------------
    # Probe first sample — detect key format + deserialization automatically
    # -----------------------------------------------------------------------
    with env.begin(write=False) as txn:
        key_fmt = _detect_key_format(txn, n_samples)

        raw = txn.get(_make_key(0, key_fmt))
        if raw is None:
            raise RuntimeError(
                f"Key 0 not found with format {key_fmt!r}. "
                "Run inspect_cache.py to see actual keys.")

        sample = _try_deserialize(raw)

        # Support both 6-tuple and other tuple lengths gracefully
        if len(sample) == 6:
            _, pose_seq, vec_seq, audio, spectrogram, _ = sample
        else:
            raise ValueError(
                f"Expected 6-element sample tuple, got {len(sample)}. "
                "Check lmdb_data_loader_expressive.py for the actual tuple layout.")

        vec_clipped  = vec_seq[0:n_poses].reshape(n_poses, -1)
        pose_clipped = pose_seq[0:n_poses].reshape(n_poses, -1)
        audio_fixed  = data_utils_expressive.make_audio_fixed_length(
            audio, expected_audio_length)
        spec_clipped = spectrogram[:, 0:expected_spectrogram_length]

    print(f'  vec_seq:      {vec_clipped.shape}  (per sample)')
    print(f'  pose_seq:     {pose_clipped.shape}')
    print(f'  audio:        {audio_fixed.shape}')
    print(f'  spectrogram:  {spec_clipped.shape}')

    os.makedirs(out_dir, exist_ok=True)

    # Allocate memmap files
    vec_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, 'vec_seq.npy'), mode='w+',
        dtype=np.float32, shape=(n_samples,) + vec_clipped.shape)
    pose_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, 'pose_seq.npy'), mode='w+',
        dtype=np.float32, shape=(n_samples,) + pose_clipped.shape)
    audio_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, 'audio.npy'), mode='w+',
        dtype=np.float32, shape=(n_samples,) + audio_fixed.shape)
    spec_mm = np.lib.format.open_memmap(
        os.path.join(out_dir, 'spectrogram.npy'), mode='w+',
        dtype=np.float32, shape=(n_samples,) + spec_clipped.shape)

    aux_list   = []
    words_list = []

    # Fill memmap files
    with env.begin(write=False) as txn:
        for i in tqdm(range(n_samples), desc='Converting'):
            raw = txn.get(_make_key(i, key_fmt))
            if raw is None:
                raise RuntimeError(
                    f"Missing key at index {i} (format={key_fmt!r}). "
                    "Cache may be incomplete.")

            sample = _try_deserialize(raw)
            word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample

            vec_mm[i]  = vec_seq[0:n_poses].reshape(n_poses, -1).astype(np.float32)
            pose_mm[i] = pose_seq[0:n_poses].reshape(n_poses, -1).astype(np.float32)
            audio_mm[i] = data_utils_expressive.make_audio_fixed_length(
                audio, expected_audio_length).astype(np.float32)
            spec_mm[i] = spectrogram[:, 0:expected_spectrogram_length].astype(np.float32)

            aux_list.append(aux_info)
            words_list.append(word_seq)

    # Flush
    del vec_mm, pose_mm, audio_mm, spec_mm
    env.close()

    with open(os.path.join(out_dir, 'aux_info.pkl'), 'wb') as f:
        pickle.dump(aux_list, f)
    with open(os.path.join(out_dir, 'words.pkl'), 'wb') as f:
        pickle.dump(words_list, f)

    metadata = {
        'n_samples':                  n_samples,
        'n_poses':                    n_poses,
        'expected_audio_length':      expected_audio_length,
        'expected_spectrogram_length': expected_spectrogram_length,
    }
    with open(os.path.join(out_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f'Done. {n_samples} samples written to {out_dir}')


if __name__ == '__main__':
    convert(parse_args())
