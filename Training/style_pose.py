import os
import json
import random
import pickle
import argparse
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_clips", type=int, default=90)
    parser.add_argument("--save_dir",  type=str, default="D:/Embeddings2")
    return parser.parse_args()

# Configurations
TRAIN_MEMMAP = "D:/train_memmap_2"
VAL_MEMMAP = "D:/val_memmap_2"
TEST_MEMMAP = "D:/test_memmap_2"

POSE_DIM = 126
STYLE_DIM = 256
BOTTLENECK_DIM = 256

BATCH_SIZE = 256
LR = 0.0003
N_EPOCHS = 200
WEIGHT_DECAY = 0.0001
SEED = 42

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Train
def train_epoch(model, loader, optimiser):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for vecs, labels in loader:
        vecs, labels = vecs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        _, id_loss = model(vecs, labels)
        id_loss.backward()
        optimiser.step()
        with torch.no_grad():
            feat = model.encoder(vecs)
            logits, _ = model.classifier(feat)
        total_loss += id_loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n

# Eval
@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for vecs, labels in loader:
        vecs, labels = vecs.to(DEVICE), labels.to(DEVICE)
        feat = model.encoder(vecs)
        logits, _ = model.classifier(feat)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(labels)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += len(labels)
    return total_loss / total_n, total_correct / total_n

# Main
def main():
    args = parse_args()
    MIN_CLIPS = args.min_clips
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nRunning with MIN_CLIPS={MIN_CLIPS}, SAVE_DIR={SAVE_DIR}")

    # Speaker index
    print("\nBuilding speaker index")
    speaker_to_idx = build_speaker_index_memmap([TRAIN_MEMMAP, VAL_MEMMAP], min_clips=MIN_CLIPS)
    n_speakers = len(speaker_to_idx)
    with open(os.path.join(SAVE_DIR, "speaker_to_idx.json"), "w") as f:
        json.dump(speaker_to_idx, f, indent=2)

    # Dataset & splits
    print("\nLoading data")
    full_dataset = GestureStyleDatasetMemmap([TRAIN_MEMMAP, VAL_MEMMAP, TEST_MEMMAP], speaker_to_idx)

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    print("\nBuilding the Model")
    model = StyleEmbeddingModel(num_speakers=n_speakers, pose_dim=POSE_DIM, style_dim=STYLE_DIM, bottleneck_dim=BOTTLENECK_DIM).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS, eta_min=LR / 20)

    # Training
    print("\nTraining")
    best_val_acc = 0.0
    best_path = os.path.join(SAVE_DIR, "pose_style_best.pt")
    last_path = os.path.join(SAVE_DIR, "pose_style_last.pt")

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimiser)
        vl_loss, vl_acc = eval_epoch(model, val_loader)
        scheduler.step()

        is_best = vl_acc > best_val_acc
        tag = "  <- best" if is_best else ""
        print(f"Epoch {epoch:03d}/{N_EPOCHS}:  train loss {tr_loss:.4f}  acc {tr_acc*100:.1f}%  |  val loss {vl_loss:.4f}  acc {vl_acc*100:.1f}%{tag}")

        if is_best:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_acc": vl_acc, "n_speakers": n_speakers,
                "style_dim": STYLE_DIM, "bottleneck_dim": BOTTLENECK_DIM,
                "pose_dim": POSE_DIM}, best_path)

    torch.save({
        "epoch": N_EPOCHS, "model_state": model.state_dict(),
        "val_acc": vl_acc, "n_speakers": n_speakers,
        "style_dim": STYLE_DIM, "bottleneck_dim": BOTTLENECK_DIM,
        "pose_dim": POSE_DIM}, last_path)

    # Reload best checkpoint
    print("\nReloading best checkpoint")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state"])

    # Test evaluation
    test_loss, test_acc = eval_epoch(model, test_loader)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc*100:.1f}%")

if __name__ == "__main__":
    main()