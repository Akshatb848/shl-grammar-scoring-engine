"""5-fold fine-tuning of DeBERTa-v3 as a grammar-score regressor on Whisper transcripts.

Command-line version of section 6b of shl_grammar_scoring.ipynb (same settings and cache files), for long runs that
should survive outside a Jupyter kernel. The notebook is the reference copy of this code.

Uses the same KFold split as the notebook (5 folds, shuffle, seed 42, train.csv row order) so the out-of-fold
predictions can be blended with the notebook's models. Each fold is saved as soon as it finishes, so a killed run
resumes at the next unfinished fold.

Usage:  .venv\\Scripts\\python finetune_deberta.py [whisper_model] [--drop-zero] [--seed=K]     (default medium.en)
        --drop-zero removes zero-labelled clips from every training fold (validation folds are unchanged).
        --seed=K    trains seed copy K (0 = original run) into its own cache folder, for seed averaging.
Writes: cache/deberta_<whisper_model>/fold<k>.npz   with oof (val idx + preds), test preds, in-fold train preds
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DROP_ZERO = "--drop-zero" in sys.argv
# --seed=K trains an extra copy with a different random seed (K=0 is the original run); copies are averaged later.
SEED_IDX = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 0)
WHISPER_MODEL = ARGS[0] if ARGS else "medium.en"
BACKBONE = "microsoft/deberta-v3-base"
MAX_LEN, BATCH, EPOCHS, LR, HEAD_LR, WARMUP = 256, 8, 6, 3e-5, 1e-3, 0.1
SEED, N_FOLDS = 42, 5
OUT = Path("cache") / (f"deberta_{WHISPER_MODEL}{'_nozero' if DROP_ZERO else ''}"
                       + (f"_seed{SEED_IDX}" if SEED_IDX else ""))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"

# ---- data: same path resolution and transcript joining as the notebook --------------------------------------
DATA = Path("data")
train_df = pd.read_csv(next(DATA.rglob("train.csv")))
test_df = pd.read_csv(next(DATA.rglob("test.csv")))
transcripts = json.loads((Path("cache") / f"transcripts_{WHISPER_MODEL}.json").read_text())
by_split_name = {(("test" if "test" in str(Path(k).parent).lower() else "train"), Path(k).name): k for k in transcripts}


def texts(df, split):
    return [" ".join(s["text"] for s in transcripts[by_split_name[(split, f)]]["segments"]).strip()
            for f in df.filename]


X_text, X_test_text = texts(train_df, "train"), texts(test_df, "test")
y = train_df.label.values.astype(np.float32)
TRAIN_MASK = (y > 0) if DROP_ZERO else np.ones(len(y), bool)
tok = AutoTokenizer.from_pretrained(BACKBONE)


def loader(text_list, labels=None, shuffle=False):
    enc = tok(text_list, truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              **({"labels": float(labels[i])} if labels is not None else {})} for i in range(len(text_list))]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch], return_tensors="pt")
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    return DataLoader(items, batch_size=BATCH if shuffle else BATCH * 4, shuffle=shuffle, collate_fn=collate)


# ---- model: backbone + attention-masked mean pooling + linear head ------------------------------------------
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE)
        self.backbone.gradient_checkpointing_enable()
        # The 128k-token embedding matrix is ~half the parameters; freezing it keeps AdamW state inside 4 GB VRAM
        # and is a mild regulariser for a 769-sample dataset.
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data.fill_(float(y[TRAIN_MASK].mean()))  # start at the label mean instead of 0

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    preds = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds.append(model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE)).float().cpu())
    return torch.cat(preds).numpy()


rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr, va) in enumerate(kf.split(np.zeros(len(y)))):
    tr = tr[TRAIN_MASK[tr]]
    out_file = OUT / f"fold{fold}.npz"
    if out_file.exists():
        print(f"fold {fold}: cached", flush=True)
        continue
    torch.manual_seed(SEED + fold + 1000 * SEED_IDX)
    np.random.seed(SEED + fold + 1000 * SEED_IDX)
    t0 = time.time()

    model = Regressor().to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=0.01)
    train_dl = loader([X_text[i] for i in tr], y[tr], shuffle=True)
    val_dl = loader([X_text[i] for i in va])
    steps = EPOCHS * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(WARMUP * steps), steps)

    # Fixed epoch budget (no early stopping on the validation fold) keeps the OOF estimate honest.
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for b in train_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE))
            loss = nn.functional.mse_loss(pred.float(), b["labels"].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val_pred = np.clip(predict(model, val_dl), 0, 5)
        print(f"fold {fold} epoch {epoch + 1}/{EPOCHS}: train_mse={np.mean(losses):.3f} "
              f"val_rmse={rmse(y[va], val_pred):.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    np.savez(out_file, val_idx=va, val_pred=val_pred, train_idx=tr,
             train_pred=np.clip(predict(model, loader([X_text[i] for i in tr])), 0, 5),
             test_pred=np.clip(predict(model, loader(X_test_text)), 0, 5))
    print(f"fold {fold}: saved, val_rmse={rmse(y[va], val_pred):.4f}", flush=True)
    del model, opt
    torch.cuda.empty_cache()

# ---- summary over all folds ---------------------------------------------------------------------------------
oof = np.zeros(len(y))
for f in range(N_FOLDS):
    d = np.load(OUT / f"fold{f}.npz")
    oof[d["val_idx"]] = d["val_pred"]
r = np.corrcoef(oof, y)[0, 1]
nz = y > 0
print(f"DeBERTa OOF: rmse={rmse(y, oof):.4f} pearson={r:.4f} | non-zero clips: rmse={rmse(y[nz], oof[nz]):.4f}",
      flush=True)
print("complete", flush=True)
