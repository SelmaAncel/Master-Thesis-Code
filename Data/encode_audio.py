"""
extract_ast_embeddings.py
=========================
Extracts 768-dim AST embeddings from anonymized wav files.
Uses the EXACT same ASTModel, config, and spectrogram preprocessing
as the original training code (AudioStyleEmbeddingModel).

Outputs:
  train_memmap_2/ast_embeddings_anon.npy  (N_train, 768)
  val_memmap_2/ast_embeddings_anon.npy    (N_val,   768)

Usage:
    python extract_ast_embeddings.py \
        --wav_scp    /home/u105121/kaldi_data/my_dataset_train_mcadams/wav.scp \
        --aux_info   /home/u105121/train_memmap_2/aux_info.pkl \
        --output_npy /home/u105121/train_memmap_2/ast_embeddings_anon.npy
"""

import os
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast

import timm
from timm.models.layers import to_2tuple, trunc_normal_

assert timm.__version__ == '0.4.5', \
    f'Need timm==0.4.5, got {timm.__version__}. Run: pip install timm==0.4.5'

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
AST_CKPT    = '/home/u105121/ast/audioset_10_10_0.4593.pth'
FREQ_BINS   = 128
TARGET_TIME = 512
SAMPLE_RATE = 12952


# ──────────────────────────────────────────────
# AST model
# ──────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size    = to_2tuple(img_size)
        patch_size  = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ASTModel(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=512,
                 imagenet_pretrain=True, model_size='base384', verbose=True):
        super().__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5'

        if verbose:
            print('---------------AST Model Summary---------------')
            print(f'ImageNet pretraining: {imagenet_pretrain}')

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.v = timm.create_model('vit_deit_base_distilled_patch16_384',
                                   pretrained=imagenet_pretrain)

        self.original_num_patches   = self.v.patch_embed.num_patches
        self.oringal_hw             = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim)
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches  = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        if verbose:
            print(f'frequency stride={fstride}, time stride={tstride}')
            print(f'number of patches={num_patches}')

        new_proj = nn.Conv2d(1, self.original_embedding_dim,
                             kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain:
            new_proj.weight = nn.Parameter(
                torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(1, self.original_embedding_dim,
                         self.oringal_hw, self.oringal_hw)
            )
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[
                    :, :, :,
                    int(self.oringal_hw/2) - int(t_dim/2):
                    int(self.oringal_hw/2) - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = F.interpolate(
                    new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[
                    :, :,
                    int(self.oringal_hw/2) - int(f_dim/2):
                    int(self.oringal_hw/2) - int(f_dim/2) + f_dim, :]
            else:
                new_pos_embed = F.interpolate(
                    new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(
                1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(),
                           new_pos_embed], dim=1))
        else:
            new_pos_embed = nn.Parameter(
                torch.zeros(1, self.v.patch_embed.num_patches + 2,
                            self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim, input_tdim):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj  = nn.Conv2d(1, self.original_embedding_dim,
                               kernel_size=(16, 16), stride=(fstride, tstride))
        test_out   = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]

    @autocast()
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return (x[:, 0] + x[:, 1]) / 2   # (B, 768)


# ──────────────────────────────────────────────
# Spectrogram
# ──────────────────────────────────────────────

def wav_to_spec(waveform, sample_rate, freq_bins=128, target_time=512):
    """
    Matches SpectrogramDataset.__getitem__ preprocessing:
      1. Compute mel spectrogram
      2. Per-frequency normalisation (mean/std)
      3. Interpolate to (freq_bins, target_time)
      4. Transpose to (target_time, freq_bins)  ← AST input format
    """
    # Build mel transform matching your sample rate
    hop_length = max(1, int(sample_rate * 2.8 / target_time))
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=freq_bins,
        f_min=50,
        f_max=sample_rate // 2,
    )

    spec = mel_transform(waveform.unsqueeze(0))          # (1, 128, T)
    spec = torch.log(spec + 1e-7).squeeze(0)              # (128, T)

    # Per-frequency normalisation (mirrors your code)
    mean = spec.mean(dim=-1, keepdim=True)
    std  = spec.std(dim=-1,  keepdim=True) + 1e-6
    spec = (spec - mean) / std                            # (128, T)

    # Interpolate to (freq_bins, target_time)
    spec = spec.unsqueeze(0).unsqueeze(0)                 # (1,1,128,T)
    spec = F.interpolate(spec, size=(freq_bins, target_time),
                         mode='bilinear', align_corners=False)
    spec = spec.squeeze(0).squeeze(0)                     # (128, 512)

    # Transpose to match AST forward() input: (TARGET_TIME, FREQ_BINS)
    spec = spec.transpose(0, 1)                           # (512, 128)
    return spec


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class AnonWavDataset(torch.utils.data.Dataset):
    def __init__(self, wav_scp, aux_info, sample_rate=12952,
                 freq_bins=128, target_time=512):
        self.utt2path = {}
        with open(wav_scp) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    self.utt2path[parts[0]] = parts[1]

        self.aux_info    = aux_info
        self.sample_rate = sample_rate
        self.freq_bins   = freq_bins
        self.target_time = target_time
        self.silence     = torch.zeros(target_time, freq_bins)

        self.entries = []
        missing = 0
        for i, meta in enumerate(aux_info):
            vid    = meta.get('vid', 'unknown')
            start  = meta.get('start_frame_no', i)
            utt_id = f'{vid}_{i:07d}_{start}'
            path   = self.utt2path.get(utt_id, None)
            self.entries.append((utt_id, path))
            if path is None:
                missing += 1
        if missing > 0:
            print(f'  WARNING: {missing:,} clips not in wav.scp (silence used)')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        utt_id, path = self.entries[idx]
        if path is None or not os.path.exists(path):
            return self.silence, idx
        try:
            waveform, sr = torchaudio.load(path)
            waveform = waveform.mean(0)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate)
            spec = wav_to_spec(waveform, self.sample_rate,
                               self.freq_bins, self.target_time)
        except Exception as e:
            print(f'  WARNING: {path}: {e}')
            spec = self.silence
        return spec, idx


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wav_scp',    required=True)
    p.add_argument('--aux_info',   required=True)
    p.add_argument('--output_npy', required=True)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers',type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice : {device}')
    print(f'wav.scp: {args.wav_scp}')
    print(f'Output : {args.output_npy}')

    # ── Load aux_info ─────────────────────────
    print('\n[1/4] Loading aux_info...')
    with open(args.aux_info, 'rb') as f:
        aux_list = pickle.load(f)
    N = len(aux_list)
    print(f'  {N:,} clips')

    # ── Load AST (exact same way as AudioStyleEmbeddingModel) ─
    print('\n[2/4] Loading AST...')
    model = ASTModel(
        label_dim=527,
        input_fdim=FREQ_BINS,
        input_tdim=TARGET_TIME,
        imagenet_pretrain=True,
        model_size='base384',
        verbose=True,
    ).to(device)

    ckpt = torch.load(AST_CKPT, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f'  Missing={len(missing)}  Unexpected={len(unexpected)}')
    model.eval()
    print(f'  Checkpoint loaded: {AST_CKPT}')

    # ── Dataset ───────────────────────────────
    print('\n[3/4] Building dataset...')
    dataset = AnonWavDataset(
        wav_scp=args.wav_scp,
        aux_info=aux_list,
        sample_rate=SAMPLE_RATE,
        freq_bins=FREQ_BINS,
        target_time=TARGET_TIME,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'  {len(dataset):,} clips | {len(loader)} batches')

    # ── Extract ───────────────────────────────
    print('\n[4/4] Extracting embeddings...')
    embeddings = np.zeros((N, 768), dtype=np.float32)

    with torch.no_grad():
        for specs, indices in tqdm(loader, desc='Extracting'):
            specs = specs.to(device, non_blocking=True)
            embs  = model(specs).float().cpu().numpy()
            for b, idx in enumerate(indices.numpy()):
                embeddings[idx] = embs[b]

    # ── Save ──────────────────────────────────
    out_path = Path(args.output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), embeddings)
    print(f'\n  Saved: {out_path}  shape={embeddings.shape}')
    print(f"  Load:  np.load('{out_path}', mmap_mode='r')")


if __name__ == '__main__':
    main()