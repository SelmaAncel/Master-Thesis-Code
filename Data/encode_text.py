import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# ── CONFIGURE ────────────────────────────────────────────────────────────────
WORDS_PKL   = "D:/train_memmap_2/words.pkl"
AUX_PKL     = "D:/train_memmap_2/aux_info.pkl"
SAVE_DIR    = "D:/EmbeddingsBERT"
BERT_MODEL  = "bert-base-uncased"

BATCH_SIZE  = 256          # clips per BERT forward pass
MAX_TOKENS  = 64           # max tokens per clip (most clips are short phrases)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


def words_to_sentence(word_list):
    """
    word_list is a list of [word, start_time, end_time] entries.
    Returns a single whitespace-joined string, or '' if empty.
    """
    tokens = []
    for entry in word_list:
        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            w = entry[0]
            if isinstance(w, str) and w.strip():
                tokens.append(w.strip())
    return " ".join(tokens)


def batched_bert_encode(sentences, tokenizer, model, batch_size, max_tokens, device):
    """
    Encode a list of strings with BERT.
    Returns np.float32 array of shape (N, 768) — [CLS] token embeddings.
    Silent clips (empty string) get a zero vector.
    """
    N = len(sentences)
    embeddings = np.zeros((N, 768), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            if start % 10000 == 0:
                print(f"  Encoding clips {start:,} / {N:,} ...", end="\r")

            batch_sentences = sentences[start : start + batch_size]

            # Separate silent clips (empty string) from non-silent
            non_empty_idx = [j for j, s in enumerate(batch_sentences) if s.strip()]
            non_empty_txt = [batch_sentences[j] for j in non_empty_idx]

            if not non_empty_txt:
                continue  # all silent — leave as zeros

            encoded = tokenizer(
                non_empty_txt,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            output = model(**encoded)
            cls_embs = output.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)

            for local_j, global_j in enumerate(non_empty_idx):
                embeddings[start + global_j] = cls_embs[local_j]

    print(f"  Encoding clips {N:,} / {N:,} ... done")
    return embeddings


def main():
    print(f"Using device: {DEVICE}")

    # ── Load memmap files ─────────────────────────────────────────────────────
    print("\nLoading words.pkl ...")
    with open(WORDS_PKL, "rb") as f:
        words = pickle.load(f)
    print(f"  {len(words):,} clip word lists")

    print("Loading aux_data.pkl ...")
    with open(AUX_PKL, "rb") as f:
        aux = pickle.load(f)
    print(f"  {len(aux):,} clip metadata entries")

    assert len(words) == len(aux), (
        f"Length mismatch: words={len(words)}, aux={len(aux)}"
    )
    N = len(words)

    # ── Build sentences ───────────────────────────────────────────────────────
    print("\nBuilding sentences from word lists ...")
    sentences = [words_to_sentence(words[i]) for i in range(N)]

    n_silent  = sum(1 for s in sentences if not s.strip())
    n_speech  = N - n_silent
    print(f"  Clips with speech : {n_speech:,}  ({100*n_speech/N:.1f}%)")
    print(f"  Silent clips      : {n_silent:,}  ({100*n_silent/N:.1f}%)")
    print(f"  Sample sentence 0 : {sentences[0]!r}")
    print(f"  Sample sentence 4 : {sentences[4]!r}")

    # ── Load BERT ─────────────────────────────────────────────────────────────
    print(f"\nLoading BERT ({BERT_MODEL}) ...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model     = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BERT parameters   : {n_params:,}  (frozen)")

    # ── Encode ────────────────────────────────────────────────────────────────
    print("\nEncoding clips with BERT ...")
    embeddings = batched_bert_encode(
        sentences, tokenizer, model,
        batch_size=BATCH_SIZE,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
    )
    print(f"  Embedding matrix  : {embeddings.shape}  dtype={embeddings.dtype}")

    # ── Build lookup dict keyed by (vid, start_frame_no) ─────────────────────
    # This is the key used at training time to look up the embedding
    # for a given LMDB clip, which also has meta['vid'] and meta['start_frame_no']
    print("\nBuilding (vid, start_frame_no) → embedding lookup ...")
    lookup = {}
    duplicates = 0

    for i, meta in enumerate(aux):
        key = (meta["vid"], int(meta["start_frame_no"]))
        if key in lookup:
            duplicates += 1
        lookup[key] = embeddings[i]

    print(f"  Unique keys       : {len(lookup):,}")
    if duplicates:
        print(f"  ⚠ Duplicate keys (overwritten): {duplicates:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(SAVE_DIR) / "bert_clip_embeddings.pkl"
    print(f"\nSaving lookup → {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  File size         : {size_mb:.1f} MB")

    # ── Save summary metadata ─────────────────────────────────────────────────
    sample_keys = [str(k) for k in list(lookup.keys())[:5]]
    meta_out = {
        "total_clips"     : N,
        "clips_with_speech": n_speech,
        "silent_clips"    : n_silent,
        "coverage_pct"    : round(100 * n_speech / N, 2),
        "embedding_dim"   : 768,
        "bert_model"      : BERT_MODEL,
        "key_format"      : "(vid: str, start_frame_no: int)",
        "sample_keys"     : sample_keys,
        "duplicates_overwritten": duplicates,
    }
    meta_path = Path(SAVE_DIR) / "bert_clip_embeddings_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  Summary metadata  → {meta_path}")

    print("\nDone.")
    print(f"  Load at training time with:")
    print(f"    import pickle")
    print(f"    with open(r'{out_path}', 'rb') as f:")
    print(f"        bert_lookup = pickle.load(f)")
    print(f"    emb = bert_lookup.get((vid, start_frame_no), np.zeros(768, dtype=np.float32))")


if __name__ == "__main__":
    main()