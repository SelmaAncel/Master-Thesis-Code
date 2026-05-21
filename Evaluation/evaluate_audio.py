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
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
CHECKPOINT = "D:/EmbeddingsAudio/poseaudio_30/pose_audio_style_best.pt"
SPEAKER_MAP = "D:/EmbeddingsAudio/poseaudio_30/speaker_to_idx.json"
TRAIN_MEMMAP = "D:/train_memmap_2"
VAL_MEMMAP = "D:/val_memmap_2"
TEST_MEMMAP = "D:/test_memmap_2"
AUDIO_TRAIN_NPY = "D:/train_memmap_2/ast_embeddings_anon.npy"
AUDIO_VAL_NPY = "D:/val_memmap_2/ast_embeddings_anon.npy"
AUDIO_TEST_NPY = "D:/test_memmap_2/ast_embeddings_anon.npy"

MAX_UNSEEN_CLIPS = 20
SAVE_DIR = "D:/eval_results/pose_audio/2s"

UNSEEN_LIST_PATH = "D:/consistent_unseen_speakers.json"

POSE_DIM = 126
FUSED_DIM = 256
AUDIO_DIM = 768
BOTTLENECK_DIM = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

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
    output_shape: (N, fused_dim)
    '''
    def __init__(self, style_dim=FUSED_DIM, input_channels=POSE_DIM,
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

# Audio encoder
class AudioEncoder(nn.Module):
    def __init__(self, audio_dim=AUDIO_DIM, fused_dim=FUSED_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(audio_dim, 512),
            nn.ReLU(),
            nn.Linear(512, fused_dim))

    def forward(self, audio_emb):
        return self.proj(audio_emb)

# Classifier
class Classifier(nn.Module):
    def __init__(self, feat_dim=FUSED_DIM * 2, num_speakers=1, hidden_dim=512, bottleneck_dim=BOTTLENECK_DIM):
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
    def __init__(self, num_speakers, pose_dim=POSE_DIM, fused_dim=FUSED_DIM, audio_dim=AUDIO_DIM, bottleneck_dim=BOTTLENECK_DIM, p=0):
        super().__init__()
        self.pose_encoder = PoseStyleEncoder(input_channels=pose_dim, style_dim=fused_dim, p=p)
        self.audio_encoder = AudioEncoder(audio_dim=audio_dim, fused_dim=fused_dim)
        self.classifier = Classifier(feat_dim=fused_dim * 2, num_speakers=num_speakers, bottleneck_dim=bottleneck_dim)

    def forward(self, pose, audio_emb, label):
        fused = torch.cat([self.pose_encoder(pose), self.audio_encoder(audio_emb)], dim=1)
        logits, bottleneck = self.classifier(fused)
        id_loss = F.cross_entropy(logits, label)
        return bottleneck, id_loss

    @torch.no_grad()
    def get_embedding(self, pose, audio_emb):
        fused = torch.cat([self.pose_encoder(pose), self.audio_encoder(audio_emb)], dim=1)
        _, bottleneck  = self.classifier(fused)
        return bottleneck

# Dataset
class GestureAudioDataset(Dataset):
    def __init__(self, memmap_dirs, audio_npy_paths, speaker_to_idx):
        if isinstance(memmap_dirs, str):
            memmap_dirs = [memmap_dirs]
            audio_npy_paths = [audio_npy_paths]
        self.samples = []
        counts = defaultdict(int)
        for memmap_dir, audio_path in zip(memmap_dirs, audio_npy_paths):
            with open(os.path.join(memmap_dir, "aux_info.pkl"), "rb") as f:
                aux_info = pickle.load(f)
            vec_seq = np.load(os.path.join(memmap_dir, "vec_seq.npy"), mmap_mode="r")
            audio_emb = np.load(audio_path, mmap_mode="r")
            print(f"Loaded memmap {memmap_dir}: {len(aux_info)} entries, shape {vec_seq.shape}, audio {audio_emb.shape}")
            for i, entry in enumerate(aux_info):
                vid = entry["vid"]
                if vid not in speaker_to_idx:
                    continue
                spk_idx = speaker_to_idx[vid]
                vec = vec_seq[i].astype(np.float32).copy()

                # Exclude empty audio embeddings
                a_emb = audio_emb[i]
                if np.abs(a_emb).sum() < 1e-6:
                    continue

                audio = a_emb.astype(np.float32).copy()
                self.samples.append((vec, spk_idx, audio))
                counts[spk_idx] += 1
        n_spk = len(set(s[1] for s in self.samples))
        print(f"Loaded {len(self.samples):,} clips from {n_spk} speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vec, spk, audio = self.samples[idx]
        return (torch.from_numpy(vec), torch.tensor(spk, dtype=torch.long), torch.from_numpy(audio))

# Function for creating subsets of the dataset
class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Embed seen data
def get_seen_embeddings(model, dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    seen_embs, seen_labels = [], []
    model.eval()
    for vecs, labels, audio_embs in loader:
        seen_embs.append(model.get_embedding(vecs.to(DEVICE), audio_embs.to(DEVICE)).cpu())
        seen_labels.append(labels)
    embs = torch.cat(seen_embs).numpy()
    labels = torch.cat(seen_labels).numpy()
    return embs, labels

# Embed unseen data
def get_unseen_embeddings(model, unseen_clips):
    result = {}
    model.eval()
    for vid, clips in unseen_clips.items():
        vecs = torch.tensor(np.stack([c[0] for c in clips]), dtype=torch.float32)
        audio_embs = torch.tensor(np.stack([c[1] for c in clips]), dtype=torch.float32)
        embs = model.get_embedding(vecs.to(DEVICE), audio_embs.to(DEVICE)).cpu().numpy()
        result[vid] = embs
    return result

# Load unseen clips
def load_unseen_clips(memmap_dirs, audio_npy_paths, unseen_vids, max_clips_per_speaker):
    clips = defaultdict(list)
    counts = defaultdict(int)
    for d, audio_path in zip(memmap_dirs, audio_npy_paths):
        aux = pickle.load(open(os.path.join(d, "aux_info.pkl"), "rb"))
        vecs = np.load(os.path.join(d, "vec_seq.npy"), mmap_mode="r")
        audio_emb = np.load(audio_path, mmap_mode="r")
        for i, entry in enumerate(aux):
            vid = entry["vid"]
            if vid not in unseen_vids:               
                continue
            if counts[vid] >= max_clips_per_speaker: 
                continue
            a_emb = audio_emb[i]
            if np.abs(a_emb).sum() < 1e-6:          
                continue 
            clips[vid].append((vecs[i].astype(np.float32).copy(), a_emb.astype(np.float32).copy()))
            counts[vid] += 1
    clips = {v: c for v, c in clips.items() if len(c) >= 2}
    print(f"Unseen: {len(clips)} speakers, {sum(len(c) for c in clips.values())} clips total")
    return clips

# Seen speaker evaluation
def seen_precision(embs, labels):
    N = len(labels)
    sims = cosine_similarity(embs, embs)
    # Exclude self
    np.fill_diagonal(sims, -2.0)
    nn_idx = np.argmax(sims, axis=1)
    correct = (labels[nn_idx] == labels).sum()
    p1 = correct / N
    print(f"[Seen] P@1 = {p1*100:.1f}%")
    return float(p1)

def seen_mean_intra(embs, labels):
    sims = cosine_similarity(embs, embs)
    same_mask = (labels[:, None] == labels[None, :])
    # Exclude self
    np.fill_diagonal(same_mask, False)
    mean = sims[same_mask].mean()
    print(f"[Seen] Mean intra-speaker cosine sim = {mean:.4f}")
    return float(mean)

def seen_mean_inter(embs, labels):
    sims = cosine_similarity(embs, embs)
    diff_mask = (labels[:, None] != labels[None, :])
    mean = sims[diff_mask].mean()
    print(f"[Seen] Mean inter-speaker cosine sim = {mean:.4f}")
    return float(mean)

# Unseen speaker evaluation
def build_pool(speaker_embs):
    pool_embs, pool_vids = [], []
    for vid, embs in sorted(speaker_embs.items()):
        pool_embs.append(embs)
        pool_vids.extend([vid] * len(embs))
    return np.concatenate(pool_embs, axis=0), np.array(pool_vids)

def unseen_precision(speaker_embs):
    pool_embs, pool_vids = build_pool(speaker_embs)
    sims = cosine_similarity(pool_embs, pool_embs)
    # Exclude self
    np.fill_diagonal(sims, -2.0)
    nn_idx = np.argmax(sims, axis=1)
    correct = (pool_vids[nn_idx] == pool_vids).sum()
    p1 = correct / len(pool_vids)
    print(f"[Unseen] P@1 = {p1*100:.1f}%")
    return float(p1)

def unseen_mean_intra(speaker_embs):
    pool_embs, pool_vids = build_pool(speaker_embs)
    sims = cosine_similarity(pool_embs, pool_embs)
    same_mask = (pool_vids[:, None] == pool_vids[None, :])
    # Exclude self
    np.fill_diagonal(same_mask, False)
    mean = sims[same_mask].mean()
    print(f"[Unseen] Mean intra-speaker cosine sim = {mean:.4f}")
    return float(mean)

def unseen_mean_inter(speaker_embs):
    pool_embs, pool_vids = build_pool(speaker_embs)
    sims = cosine_similarity(pool_embs, pool_embs)
    diff_mask = (pool_vids[:, None] != pool_vids[None, :])
    mean = sims[diff_mask].mean()
    print(f"[Unseen] Mean inter-speaker cosine sim = {mean:.4f}")
    return float(mean)

# Main
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load model
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = StyleEmbeddingModel(num_speakers = ckpt["n_speakers"], pose_dim = ckpt.get("pose_dim", POSE_DIM), fused_dim = ckpt.get("fused_dim", FUSED_DIM), audio_dim = ckpt.get("audio_dim", AUDIO_DIM), bottleneck_dim = ckpt.get("bottleneck_dim", BOTTLENECK_DIM)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"n_speakers={ckpt['n_speakers']}, val_acc={float(ckpt.get('val_acc', 0))*100:.1f}%")

    # Load dataset
    speaker_to_idx = json.load(open(SPEAKER_MAP))
    print("\nLoading dataset")
    full_dataset = GestureAudioDataset(memmap_dirs = [TRAIN_MEMMAP, VAL_MEMMAP, TEST_MEMMAP], audio_npy_paths = [AUDIO_TRAIN_NPY, AUDIO_VAL_NPY, AUDIO_TEST_NPY], speaker_to_idx = speaker_to_idx)

    speaker_to_indices = defaultdict(list)
    for idx, (_, spk, _) in enumerate(full_dataset.samples):
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

    # Seen evaluation
    print("\nSeen speaker evaluation")
    embs, labels = get_seen_embeddings(model, test_dataset)
    seen_precision(embs, labels)
    seen_mean_intra(embs, labels)
    seen_mean_inter(embs, labels)

    # Unseen evaluation
    print("\nUnseen speaker evaluation")
    unseen_vids = set(json.load(open(UNSEEN_LIST_PATH))["speakers"])
    unseen_clips = load_unseen_clips([TRAIN_MEMMAP, VAL_MEMMAP, TEST_MEMMAP], [AUDIO_TRAIN_NPY, AUDIO_VAL_NPY, AUDIO_TEST_NPY], unseen_vids, MAX_UNSEEN_CLIPS)
    speaker_embs = get_unseen_embeddings(model, unseen_clips)

    unseen_precision(speaker_embs)
    unseen_mean_intra(speaker_embs)
    unseen_mean_inter(speaker_embs)

if __name__ == "__main__":
    main()