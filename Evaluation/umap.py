import os
import json
import pickle
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Path to list with unseen speakers
UNSEEN_LIST_PATH = "D:/consistent_unseen_speakers.json"

# Clip length for evaluation
DURATION = "2s"

# Configurations for each clip lenght
DURATION_CONFIGS = {
    "2s": {
        "checkpoint": "D:/Embeddings2/pose_30/mixstage_style_best.pt",
        "speaker_map": "D:/Embeddings2/pose_30/speaker_to_idx.json",
        "memmaps": ["D:/train_memmap_2", "D:/val_memmap_2", "D:/test_memmap_2"],
        "max_unseen_clips": 20,
        "save_dir": "D:/eval_results/2s",
    }
}

# Model parameters
POSE_DIM = 126
STYLE_DIM = 256
BOTTLENECK_DIM = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ConvNormRelu from Ahuja et al. (2020)
class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 type='1d', leaky=False,
                 downsample=False, kernel_size=None, stride=None,
                 padding=None, p=0, groups=1):
        super(ConvNormRelu, self).__init__()
        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                assert len(kernel_size) == len(stride)
                padding = tuple(int((ks - st) / 2) for ks, st in zip(kernel_size, kernel_size))
            else:
                padding = int((kernel_size - stride) / 2)

        in_channels  = in_channels  * groups
        out_channels = out_channels * groups
        if type == '1d':
            self.conv    = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=groups)
            self.norm    = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv    = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=groups)
            self.norm    = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        return self.relu(self.norm(self.dropout(self.conv(x))))

# Pose Style Encoder based on the encoder from Ahuja et al. (2020)
class PoseStyleEncoder(nn.Module):
    '''
    input_shape:  (N, time, pose_features: 126)
    output_shape: (N, style_dim)
    '''
    def __init__(self, style_dim=STYLE_DIM, input_channels=POSE_DIM,
                 kernel_size=None, stride=None, p=0, groups=1, num_speakers=4):
        super().__init__()
        self.conv = nn.ModuleList([])
        self.conv.append(ConvNormRelu(input_channels, 64,  type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64,  64,  type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(64,  128, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=True,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))
        self.conv.append(ConvNormRelu(256, style_dim, type='1d', leaky=True, downsample=False,
                                      kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    def forward(self, x, time_steps=None):
        x = torch.transpose(x, 1, 2)
        if time_steps is None:
            time_steps = x.shape[-2]
        x = nn.Sequential(*self.conv)(x)
        x = x.mean(-1)
        x = x.squeeze(dim=-1)
        return x

# Classifier
class Classifier(nn.Module):
    def __init__(self, feat_dim=STYLE_DIM, num_speakers=1, hidden_dim=512, bottleneck_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(bottleneck_dim, num_speakers)

    def forward(self, feat):
        x = self.relu1(self.fc1(feat))
        bottleneck = self.relu2(self.fc2(x))
        logits = self.classifier(bottleneck)
        return logits, bottleneck

# Combining the encoders and classifier
class StyleEmbeddingModel(nn.Module):
    def __init__(self, num_speakers, pose_dim=POSE_DIM, style_dim=STYLE_DIM, bottleneck_dim=BOTTLENECK_DIM, p=0):
        super().__init__()
        self.encoder = PoseStyleEncoder(input_channels=pose_dim, style_dim=style_dim, p=p)
        self.classifier = Classifier(feat_dim=style_dim, num_speakers=num_speakers, bottleneck_dim=bottleneck_dim)

    def forward(self, pose, label):
        feat = self.encoder(pose)
        logits, bottleneck = self.classifier(feat)
        id_loss = F.cross_entropy(logits, label)
        return bottleneck, id_loss

    @torch.no_grad()
    def get_embedding(self, pose):
        feat = self.encoder(pose)
        _, bottleneck = self.classifier(feat)
        return bottleneck

# Build speaker index
def build_speaker_index_memmap(memmap_dirs, min_clips=0):
    if isinstance(memmap_dirs, str):
        memmap_dirs = [memmap_dirs]
    counts = defaultdict(int)
    for memmap_dir in memmap_dirs:
        with open(os.path.join(memmap_dir, "aux_info.pkl"), "rb") as f:
            aux_info = pickle.load(f)
        for entry in aux_info:
            counts[entry["vid"]] += 1
    if min_clips > 0:
        counts = {vid: c for vid, c in counts.items() if c >= min_clips}
    speaker_to_idx = {vid: i for i, vid in enumerate(sorted(counts))}
    print(f"Speakers (>={min_clips} clips): {len(speaker_to_idx)}")
    return speaker_to_idx

# Dataset
class GestureStyleDatasetMemmap(Dataset):
    def __init__(self, memmap_dirs, speaker_to_idx):
        if isinstance(memmap_dirs, str):
            memmap_dirs = [memmap_dirs]
        self.samples = []
        counts = defaultdict(int)
        for memmap_dir in memmap_dirs:
            with open(os.path.join(memmap_dir, "aux_info.pkl"), "rb") as f:
                aux_info = pickle.load(f)
            vec_seq = np.load(os.path.join(memmap_dir, "vec_seq.npy"), mmap_mode="r")
            print(f"Loaded memmap {memmap_dir}: {len(aux_info):,} entries, shape {vec_seq.shape}")
            for i, entry in enumerate(aux_info):
                vid = entry["vid"]
                if vid not in speaker_to_idx:
                    continue
                spk_idx = speaker_to_idx[vid]
                vec = vec_seq[i].astype(np.float32).copy()
                self.samples.append((vec, spk_idx))
                counts[spk_idx] += 1
        n_spk = len(set(s[1] for s in self.samples))
        print(f"Loaded {len(self.samples):,} clips from {n_spk} speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, spk = self.samples[idx]
        return torch.from_numpy(vec), torch.tensor(spk, dtype=torch.long)

# Function for creating subsets of the dataset
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # expose subset of samples for analysis / visualization
        if hasattr(dataset, "samples"):
            self.samples = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Load unseen clips
def load_unseen_clips(memmap_dirs, fixed_unseen_vids, max_unseen_clips):
    if isinstance(memmap_dirs, str):
        memmap_dirs = [memmap_dirs]
    unseen_clips = defaultdict(list)
    clip_counts = defaultdict(int)
    for memmap_dir in memmap_dirs:
        with open(os.path.join(memmap_dir, "aux_info.pkl"), "rb") as f:
            aux_info = pickle.load(f)
        vec_seq = np.load(os.path.join(memmap_dir, "vec_seq.npy"), mmap_mode="r")
        for i, entry in enumerate(aux_info):
            vid = entry["vid"]
            if vid not in fixed_unseen_vids: continue
            if clip_counts[vid] >= max_unseen_clips: continue
            unseen_clips[vid].append(vec_seq[i].astype(np.float32).copy())
            clip_counts[vid] += 1
    unseen_clips = {vid: clips for vid, clips in unseen_clips.items() if len(clips) >= 2}
    total = sum(len(v) for v in unseen_clips.values())
    print(f"unseen speakers loaded: {len(unseen_clips)}  |  total clips: {total:,}")
    return unseen_clips

# Visualisation
def plot_unseen_umap(speaker_embs, evaluated_vids, save_dir, duration, n_neighbors=15, min_dist=0.1, seed=42):
    all_embs_list, all_vids_arr = [], []
    for vid in evaluated_vids:
        embs = speaker_embs[vid]
        all_embs_list.append(embs)
        all_vids_arr.extend([vid] * len(embs))

    all_embs = np.concatenate(all_embs_list, axis=0)
    all_vids_arr = np.array(all_vids_arr)

    print(f"Running UMAP on unseen embeddings")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed, verbose=False)
    proj = reducer.fit_transform(all_embs)

    n_speakers = len(evaluated_vids)
    cmap20 = cm.get_cmap("tab20", 20)
    vid_to_col = {v: i % 20 for i, v in enumerate(evaluated_vids)}
    colors_spk = [cmap20(vid_to_col[v]) for v in all_vids_arr]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(proj[:, 0], proj[:, 1], c=colors_spk, alpha=0.6, s=12, linewidths=0)
    ax.set_title(f"UMAP Unseen Speakers")
    ax.set_xlabel("UMAP_1", color="black")
    ax.set_ylabel("UMAP_2", color="black")

    plt.tight_layout()
    path = os.path.join(save_dir, "unseen_umap.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  UMAP saved → {path}")


def plot_seen_umap(model, dataset, save_dir, duration, n_speakers=100, clips_per_speaker=20, n_neighbors=15, min_dist=0.1, seed=42):
    speaker_to_indices = defaultdict(list)
    for idx, (_, spk) in enumerate(dataset.samples):
        speaker_to_indices[spk].append(idx)

    eligible = [spk for spk, idxs in speaker_to_indices.items() if len(idxs) >= clips_per_speaker]
    random.seed(seed)
    selected = random.sample(eligible, min(n_speakers, len(eligible)))

    all_embs_list, all_labels = [], []
    model.eval()
    for spk in selected:
        idxs = random.sample(speaker_to_indices[spk], clips_per_speaker)
        vecs = torch.stack([dataset[i][0] for i in idxs]).to(DEVICE)
        with torch.no_grad():
            embs = model.get_embedding(vecs).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        all_embs_list.append(embs / norms)
        all_labels.extend([spk] * clips_per_speaker)

    all_embs = np.concatenate(all_embs_list, axis=0)
    all_labels = np.array(all_labels)

    print(f"\n  Running UMAP on seen-speaker embeddings "
          f"({len(selected)} speakers × {clips_per_speaker} clips)...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed, verbose=False)
    proj = reducer.fit_transform(all_embs)

    cmap20 = cm.get_cmap("tab20", 20)
    spk_list = sorted(set(all_labels))
    spk_to_col = {s: i % 20 for i, s in enumerate(spk_list)}
    colors = [cmap20(spk_to_col[s]) for s in all_labels]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.6, s=12, linewidths=0)
    ax.set_title(f"UMAP Seen Speakers")
    ax.set_xlabel("UMAP_1", color="black")
    ax.set_ylabel("UMAP_2", color="black")

    plt.tight_layout()
    path = os.path.join(save_dir, "seen_umap.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  UMAP saved → {path}")

def _encode_unseen(model, unseen_clips):
    speaker_embs = {}
    for vid, clips in unseen_clips.items():
        vecs = torch.tensor(np.stack(clips), dtype=torch.float32)
        embs_list = []
        for i in range(0, len(vecs), 512):
            embs_list.append(model.get_embedding(vecs[i:i+512].to(DEVICE)).cpu())
        embs = torch.cat(embs_list, dim=0).numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        speaker_embs[vid] = embs / norms
    return speaker_embs

# Main
def main():
    cfg = DURATION_CONFIGS[DURATION]
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Load unseen speaker list
    print(f"\nLoading unseen speaker list: {UNSEEN_LIST_PATH}")
    with open(UNSEEN_LIST_PATH) as f:
        unseen_data = json.load(f)
    fixed_unseen_vids = set(unseen_data["speakers"])

    # Load checkpoint
    print(f"\nLoading checkpoint: {cfg['checkpoint']}")
    ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
    model = StyleEmbeddingModel(
        num_speakers = ckpt["n_speakers"],
        pose_dim = ckpt.get("pose_dim", POSE_DIM),
        style_dim = ckpt.get("style_dim", STYLE_DIM),
        bottleneck_dim = ckpt.get("bottleneck_dim", BOTTLENECK_DIM)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_acc = float(ckpt.get("val_acc", 0))
    print(f"n_speakers={ckpt['n_speakers']}  val_acc={val_acc*100:.1f}%")

    # Load dataset
    with open(cfg["speaker_map"]) as f:
        speaker_to_idx = json.load(f)

    print(f"\nLoading dataset")
    full_dataset = GestureStyleDatasetMemmap(cfg["memmaps"], speaker_to_idx)

    # Load speakers used in training
    speaker_to_indices = defaultdict(list)
    for idx, (_, spk) in enumerate(full_dataset.samples):
        speaker_to_indices[spk].append(idx)

    train_indices, val_indices, test_indices = [], [], []
    for spk, idxs in speaker_to_indices.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        test_indices.extend(idxs[:n_test])
        val_indices.extend(idxs[n_test:n_test + n_val])
        train_indices.extend(idxs[n_test + n_val:])

    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)
    test_dataset = SubsetDataset(full_dataset, test_indices)
    print(f"Total: {len(full_dataset)}, Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    plot_seen_umap(model, test_dataset, cfg["save_dir"], DURATION, n_speakers=50, clips_per_speaker=20, n_neighbors=15, min_dist=0.1, seed=SEED)
    unseen_clips = load_unseen_clips(cfg["memmaps"], fixed_unseen_vids = fixed_unseen_vids, max_unseen_clips = cfg["max_unseen_clips"])
    speaker_embs = _encode_unseen(model, unseen_clips)
    eval_vids = sorted(speaker_embs.keys())
    plot_unseen_umap(speaker_embs, eval_vids, cfg["save_dir"], DURATION, n_neighbors=15, min_dist=0.1, seed=SEED)

if __name__ == "__main__":
    main() 