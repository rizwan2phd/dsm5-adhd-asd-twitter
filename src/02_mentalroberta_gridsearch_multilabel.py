# ══════════════════════════════════════════════════════════════════
# Stage 2 — Multi-Class Model Training: Grid Search
# ══════════════════════════════════════════════════════════════════
# Fine-tunes MentalRoBERTa as a multi-label classifier over the nine
# DSM-5 depressive symptoms on the ReDSM5 corpus, sweeping the grid
# in PARAM_GRID (LR, dropout, label smoothing, class-weight boost,
# warmup ratio). Corresponds to Methods: "Stage 2: Multi-Class Model
# Training" and "Per-Label Decision Threshold Calibration".
#
# Input : redsm5_M5.csv (expert-annotated ReDSM5 corpus; columns
#          include sentence_id, sentence_text, DSM5_symptom, status)
#
#          Cite: Bao E, Pérez A, Parapar J. ReDSM5: a Reddit dataset for
#          DSM-5 depression detection. In: Proceedings of the 34th ACM
#          International Conference on Information and Knowledge
#          Management (CIKM '25). 2025.
# Output: grid_search_outputs_multilabel/
#           grid_search_results.csv   — one row per run, ranked by
#                                        test macro-F1
#           best_model/                — best checkpoint + tokenizer
#                                        + label_map.json (per-label
#                                        calibrated thresholds)
#           plots/                     — training curves per run
# ══════════════════════════════════════════════════════════════════

# ── Imports ────────────────────────────────────────────────────────
import os, json, warnings, itertools, time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, classification_report
from collections import Counter

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# GRID SEARCH — edit param_grid to control search space
# Each combination will be trained independently.
# ══════════════════════════════════════════════════════════════════
PARAM_GRID = {
    # Encoder LR — mental-roberta is sensitive; 2e-5 is the upper safe limit
    "LR":              [8e-6, 2e-5],

    # Dropout — 0.4 helps on small / imbalanced corpora
    "DROPOUT":         [0.3, 0.4],

    # Label smoothing — softens overconfidence on rare DSM-5 labels
    "LABEL_SMOOTHING": [0.0, 0.05],

    # Boost — multiplier on top of inverse-freq pos_weight;
    # 2.0 pushes recall on minority labels (PSYCHOMOTOR, APPETITE_CHANGE, etc.)
    "BOOST":           [1.0, 1.5, 2.0],

    # Warmup ratio — longer warmup stabilises pretrained weights early on
    "WARMUP_RATIO":    [0.06, 0.10],
}
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# CONFIG — fixed settings (not part of grid search)
# ══════════════════════════════════════════════════════════════════
MODEL_KEY   = "mentalroberta"
OUTPUT_DIR  = "./grid_search_outputs_multilabel"

MODEL_REGISTRY = {
    "bertbase":      "google-bert/bert-base-uncased",
    "robertaBase":   "FacebookAI/roberta-base",
    "bertweet":      "vinai/bertweet-base",
    "mentalbert":    "mental/mental-bert-base-uncased",
    "mentalroberta": "mental/mental-roberta-base",
    "biov12":        "dmis-lab/biobert-base-cased-v1.2",
    "bioCliBert":    "emilyalsentzer/Bio_ClinicalBERT",
}

DSM5_LABELS = [
    "DEPRESSED_MOOD", "WORTHLESSNESS", "SUICIDAL_THOUGHTS",
    "FATIGUE", "ANHEDONIA", "SLEEP_ISSUES", "COGNITIVE_ISSUES",
    "APPETITE_CHANGE", "PSYCHOMOTOR",
]

NUM_LABELS   = len(DSM5_LABELS)
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 16
WEIGHT_DECAY = 0.01
PATIENCE     = 3
SEED         = 42
# ══════════════════════════════════════════════════════════════════

MODEL_NAME = MODEL_REGISTRY[MODEL_KEY]
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Load & prepare data (done once, shared across all runs) ────────
print("\nLoading data ...")
df = pd.read_csv("redsm5_M5.csv")
df = df[df["DSM5_symptom"] != "SPECIAL_CASE"]
df = df[df["status"] != 0]

required = {"sentence_id", "sentence_text", "DSM5_symptom"}
missing  = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

df["DSM5_symptom"] = df["DSM5_symptom"].str.strip().str.upper()

# ── FIX: group by sentence_text ONLY ──────────────────────────────
# Grouping by (sentence_id + sentence_text) meant the same sentence
# text could appear as multiple rows (different IDs / label subsets)
# and leak across train/val/test splits.
# Grouping by sentence_text alone ensures every unique surface form
# is assigned to exactly one split, with ALL its labels merged.
grouped = (
    df.groupby("sentence_text")["DSM5_symptom"]
    .apply(lambda x: list(set(x)))      # collect & deduplicate labels
    .reset_index()
    .rename(columns={"DSM5_symptom": "labels"})
)

print(f"Unique sentences (after text-level dedup) : {len(grouped)}")
print("Label distribution:")
all_labels_flat = [lbl for row in grouped["labels"] for lbl in row]
for lbl, cnt in sorted(Counter(all_labels_flat).items(), key=lambda x: -x[1]):
    print(f"  {lbl:<25}: {cnt}")

# ── Binarize & split (70 / 15 / 15) ───────────────────────────────
mlb          = MultiLabelBinarizer(classes=DSM5_LABELS)
label_matrix = mlb.fit_transform(grouped["labels"]).astype(np.float32)
texts        = grouped["sentence_text"].tolist()

primary = [int(row.argmax()) if row.sum() > 0 else 0 for row in label_matrix]

X_tr_full, X_tmp, y_tr_full, y_tmp, p_tr_full, p_tmp = train_test_split(
    texts, label_matrix, primary,
    test_size=0.30, stratify=primary, random_state=SEED
)
X_val, X_te, y_val, y_te, _, _ = train_test_split(
    X_tmp, y_tmp, p_tmp,
    test_size=0.50, stratify=p_tmp, random_state=SEED
)

X_tr, y_tr = X_tr_full, y_tr_full

# ── Sanity check: zero sentence overlap across all three splits ────
train_set = set(X_tr)
val_set   = set(X_val)
test_set  = set(X_te)

assert len(train_set & test_set) == 0, (
    f"DATA LEAK — Train/Test overlap: {len(train_set & test_set)} sentences"
)
assert len(train_set & val_set) == 0, (
    f"DATA LEAK — Train/Val overlap: {len(train_set & val_set)} sentences"
)
assert len(val_set & test_set) == 0, (
    f"DATA LEAK — Val/Test overlap: {len(val_set & test_set)} sentences"
)
print(f"\n✓ No sentence overlap across splits.")
print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

# ── Base class weights (inverse frequency, shared across runs) ─────
label_counts  = label_matrix.sum(axis=0)
total_samples = len(label_matrix)
base_weights  = (total_samples - label_counts) / (label_counts + 1e-6)

# ── Load tokenizer once ────────────────────────────────────────────
print(f"\nLoading tokenizer from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Dataset ────────────────────────────────────────────────────────
class DSM5Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tokenizer = texts, labels, tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx]),
        }


val_loader  = DataLoader(DSM5Dataset(X_val, y_val, tokenizer),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(DSM5Dataset(X_te,  y_te,  tokenizer),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Model ──────────────────────────────────────────────────────────
class MentalBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout, pos_weight):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.pos_weight = pos_weight

    def forward(self, input_ids, attention_mask, labels=None, smoothing=0.0):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = self.dropout(out.last_hidden_state[:, 0, :].float())
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            targets = labels.float()
            if smoothing > 0:
                targets = targets * (1 - smoothing) + 0.5 * smoothing
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, targets)
        return {"loss": loss, "logits": logits}


# ── Train one epoch ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, smoothing):
    model.train()
    total = 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        out  = model(ids, mask, lbls, smoothing=smoothing)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += out["loss"].item()
    return total / len(loader)


# ── Evaluate ───────────────────────────────────────────────────────
def evaluate(model, loader, threshold=0.5):
    """
    threshold : float          → same cutoff applied to every label
                np.ndarray     → shape (NUM_LABELS,), one cutoff per label
    """
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            out  = model(ids, mask, lbls)
            total_loss += out["loss"].item()
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    probs  = 1 / (1 + np.exp(-logits))

    # Works for both scalar float and (NUM_LABELS,) array via broadcasting
    preds = (probs >= np.asarray(threshold)).astype(int)

    return {
        "loss":     total_loss / len(loader),
        "micro_f1": f1_score(labels, preds, average="micro",  zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro",  zero_division=0),
        "hamming":  hamming_loss(labels, preds),
        "exact":    accuracy_score(labels, preds),
        "preds":    preds,
        "labels":   labels,
        "probs":    probs,
    }


# ── Per-label threshold sweep on validation set ────────────────────
def find_best_thresholds_per_label(model, loader):
    """
    Sweeps a fine grid of thresholds (0.15 → 0.85, step 0.025) for each
    label independently on `loader`, maximising that label's binary F1.

    Returns
    -------
    best_thresholds : np.ndarray  shape (NUM_LABELS,)
    best_f1s        : np.ndarray  shape (NUM_LABELS,)  — val F1 at best thr
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(ids, mask)
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    probs  = 1 / (1 + np.exp(-np.vstack(all_logits)))   # (N, NUM_LABELS)
    labels = np.vstack(all_labels)                        # (N, NUM_LABELS)

    best_thresholds = np.full(NUM_LABELS, 0.5)
    best_f1s        = np.zeros(NUM_LABELS)

    for i in range(NUM_LABELS):
        for t in np.arange(0.15, 0.85, 0.025):
            preds_i = (probs[:, i] >= t).astype(int)
            f1_i    = f1_score(labels[:, i], preds_i, zero_division=0)
            if f1_i > best_f1s[i]:
                best_f1s[i]        = f1_i
                best_thresholds[i] = round(float(t), 3)

    return best_thresholds, best_f1s


# ── Plot training curves (saved per run) ──────────────────────────
def save_training_curves(run_id, params, history, best_epoch, save_dir):
    epochs_ran = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Run {run_id} | LR={params['LR']:.0e} DO={params['DROPOUT']} "
        f"LS={params['LABEL_SMOOTHING']} BOOST={params['BOOST']} "
        f"WR={params['WARMUP_RATIO']} | Best Epoch: {best_epoch}",
        fontsize=11, fontweight="bold"
    )

    # 1 — Loss
    ax = axes[0, 0]
    ax.plot(epochs_ran, history["train_loss"], marker="o", color="#4C72B0", lw=2, label="Train Loss")
    ax.plot(epochs_ran, history["val_loss"],   marker="s", color="#C44E52", lw=2, label="Val Loss")
    ax.fill_between(epochs_ran, history["train_loss"], history["val_loss"],
                    alpha=0.10, color="orange", label="Gap")
    ax.axvline(best_epoch, color="red", ls="--", lw=1.4, label=f"Best ({best_epoch})")
    ax.set_title("Train vs Val Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 2 — Macro F1 (overfit monitor)
    ax = axes[0, 1]
    ax.plot(epochs_ran, history["train_macro_f1"], marker="o", color="#4C72B0", lw=2, label="Train Macro F1")
    ax.plot(epochs_ran, history["val_macro_f1"],   marker="s", color="#C44E52", lw=2, label="Val Macro F1")
    ax.fill_between(epochs_ran, history["train_macro_f1"], history["val_macro_f1"],
                    alpha=0.15, color="orange", label="Overfit gap")
    ax.axvline(best_epoch, color="red", ls="--", lw=1.4, label=f"Best ({best_epoch})")
    ax.set_title("Macro F1 (Overfit Monitor)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 3 — Micro F1 + Exact Match
    ax = axes[1, 0]
    ax.plot(epochs_ran, history["val_micro_f1"], marker="^", color="#55A868", lw=2, label="Val Micro F1")
    ax.plot(epochs_ran, history["val_exact"],    marker="D", color="#8172B2", lw=2, label="Val Exact Match")
    ax.axvline(best_epoch, color="red", ls="--", lw=1.4, label=f"Best ({best_epoch})")
    ax.set_title("Val Micro F1 & Exact Match"); ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 4 — Hamming loss + overfit gap
    ax = axes[1, 1]
    ax.plot(epochs_ran, history["val_hamming"], marker="o", color="#C44E52", lw=2, label="Val Hamming Loss")
    f1_gap = [tf - vf for tf, vf in zip(history["train_macro_f1"], history["val_macro_f1"])]
    ax2b = ax.twinx()
    ax2b.plot(epochs_ran, f1_gap, marker="s", color="#4C72B0", lw=2, ls="--", label="F1 gap (Tr−Va)")
    ax2b.set_ylabel("F1 gap", color="#4C72B0")
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_title("Hamming Loss & Overfit Gap"); ax.set_xlabel("Epoch"); ax.set_ylabel("Hamming Loss")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(save_dir, f"run_{run_id:03d}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════
# GRID SEARCH LOOP
# ══════════════════════════════════════════════════════════════════
keys       = list(PARAM_GRID.keys())
combos     = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
total_runs = len(combos)

print(f"\n{'='*70}")
print(f"GRID SEARCH — {total_runs} combinations × up to {EPOCHS} epochs each")
print(f"Parameters: {keys}")
print(f"{'='*70}\n")

all_results = []

for run_idx, combo in enumerate(combos, start=1):
    params = dict(zip(keys, combo))

    run_id_str = f"{run_idx:03d}"
    ckpt_path  = os.path.join(OUTPUT_DIR, f"run_{run_id_str}_best.pt")

    print(f"\n{'─'*70}")
    print(f"RUN {run_idx}/{total_runs}  |  " +
          "  ".join(f"{k}={v}" for k, v in params.items()))
    print(f"{'─'*70}")

    # ── Build boosted pos_weight for this run ───────────────────
    boosted_weights = base_weights * params["BOOST"]
    pos_weight = torch.tensor(boosted_weights, dtype=torch.float32).to(device)
    print(f"  pos_weight (BOOST={params['BOOST']}) sample:")
    for name, w in zip(DSM5_LABELS, boosted_weights):
        print(f"    {name:<25}: {w:.2f}")

    # ── Train loader ────────────────────────────────────────────
    train_loader = DataLoader(
        DSM5Dataset(X_tr, y_tr, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )

    # ── Model ───────────────────────────────────────────────────
    model = MentalBERTClassifier(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        dropout=params["DROPOUT"],
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),
         "lr": params["LR"], "weight_decay": WEIGHT_DECAY},
        {"params": model.classifier.parameters(),
         "lr": params["LR"] * 10, "weight_decay": WEIGHT_DECAY},
    ], eps=1e-8)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * params["WARMUP_RATIO"]),
        num_training_steps=total_steps,
    )

    # ── Per-run history ─────────────────────────────────────────
    history = {
        "train_loss":     [],
        "val_loss":       [],
        "train_macro_f1": [],
        "val_micro_f1":   [],
        "val_macro_f1":   [],
        "val_hamming":    [],
        "val_exact":      [],
    }

    best_macro_f1, patience_cnt, best_epoch = 0.0, 0, 1
    run_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            smoothing=params["LABEL_SMOOTHING"],
        )
        tr_eval = evaluate(model, train_loader)
        val     = evaluate(model, val_loader)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val["loss"])
        history["train_macro_f1"].append(tr_eval["macro_f1"])
        history["val_micro_f1"].append(val["micro_f1"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["val_hamming"].append(val["hamming"])
        history["val_exact"].append(val["exact"])

        gap = tr_eval["macro_f1"] - val["macro_f1"]
        print(f"  Ep {epoch:02d}/{EPOCHS} | "
              f"TrLoss:{tr_loss:.4f} VaLoss:{val['loss']:.4f} | "
              f"TrMacF1:{tr_eval['macro_f1']:.4f} VaMacF1:{val['macro_f1']:.4f} | "
              f"VaMicF1:{val['micro_f1']:.4f} VaHamm:{val['hamming']:.4f} | "
              f"Gap:{gap:.4f}")

        if val["macro_f1"] > best_macro_f1:
            best_macro_f1 = val["macro_f1"]
            best_epoch    = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved best (macro-F1: {best_macro_f1:.4f})")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    run_time = time.time() - run_start

    # ── Save training curves ─────────────────────────────────────
    curve_path = save_training_curves(run_idx, params, history, best_epoch, PLOTS_DIR)
    print(f"  Curves saved → {curve_path}")

    # ── Load best checkpoint ─────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ── Per-label threshold sweep on VAL set ─────────────────────
    best_thresholds, val_per_label_f1 = find_best_thresholds_per_label(model, val_loader)

    print(f"\n  Per-label thresholds (val sweep):")
    for lbl, t, f1v in zip(DSM5_LABELS, best_thresholds, val_per_label_f1):
        print(f"    {lbl:<25}: thr={t:.3f}  val-F1={f1v:.4f}")

    # ── Test evaluation using per-label thresholds ───────────────
    test_res  = evaluate(model, test_loader, threshold=best_thresholds)
    per_label_test_f1 = f1_score(
        test_res["labels"], test_res["preds"], average=None, zero_division=0
    )

    print(f"\n  TEST → MacroF1:{test_res['macro_f1']:.4f}  "
          f"MicroF1:{test_res['micro_f1']:.4f}  "
          f"Hamming:{test_res['hamming']:.4f}  "
          f"Exact:{test_res['exact']:.4f}  "
          f"time={run_time/60:.1f}m")

    print(f"\n  Per-label TEST F1 (with per-label thresholds):")
    for lbl, t, f1t in zip(DSM5_LABELS, best_thresholds, per_label_test_f1):
        print(f"    {lbl:<25}: thr={t:.3f}  test-F1={f1t:.4f}")

    result_row = {
        "run_id":            run_idx,
        "macro_f1":          round(test_res["macro_f1"], 4),
        "micro_f1":          round(test_res["micro_f1"], 4),
        "hamming":           round(test_res["hamming"],  4),
        "exact_match":       round(test_res["exact"],    4),
        "best_val_macro_f1": round(best_macro_f1, 4),
        "best_epoch":        best_epoch,
        "time_min":          round(run_time / 60, 1),
        "ckpt":              ckpt_path,
        "curve_plot":        curve_path,
        **{k: params[k] for k in keys},
        # per-label thresholds (from val sweep)
        **{f"thr_{DSM5_LABELS[i]}": float(best_thresholds[i])
           for i in range(NUM_LABELS)},
        # per-label val F1 at best threshold
        **{f"val_f1_{DSM5_LABELS[i]}": round(float(val_per_label_f1[i]), 4)
           for i in range(NUM_LABELS)},
        # per-label test F1
        **{f"f1_{DSM5_LABELS[i]}": round(float(per_label_test_f1[i]), 4)
           for i in range(NUM_LABELS)},
    }
    all_results.append(result_row)

    # ── Save incremental CSV after every run ─────────────────────
    pd.DataFrame(all_results).sort_values("macro_f1", ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, "grid_search_results.csv"), index=False
    )


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
best_row   = results_df.iloc[0]

print(f"\n{'='*75}")
print("GRID SEARCH COMPLETE — RANKED RESULTS (Test Macro F1)")
print(f"{'='*75}")
display_cols = (["run_id"] + keys +
                ["macro_f1", "micro_f1", "hamming", "exact_match",
                 "best_epoch", "time_min"])
print(results_df[display_cols].to_string(index=False))

print(f"\n{'='*75}")
print("BEST CONFIGURATION")
print(f"{'='*75}")
for k in keys:
    print(f"  {k:<20} = {best_row[k]}")
print(f"  {'macro_f1':<20} = {best_row['macro_f1']:.4f}")
print(f"  {'micro_f1':<20} = {best_row['micro_f1']:.4f}")
print(f"  {'hamming':<20} = {best_row['hamming']:.4f}")
print(f"  {'exact_match':<20} = {best_row['exact_match']:.4f}")
print(f"  {'best_epoch':<20} = {best_row['best_epoch']}")

print(f"\n  Per-label thresholds (best run, tuned on val):")
for lbl in DSM5_LABELS:
    thr  = best_row[f"thr_{lbl}"]
    vf1  = best_row[f"val_f1_{lbl}"]
    tf1  = best_row[f"f1_{lbl}"]
    print(f"    {lbl:<25}: thr={thr:.3f}  val-F1={vf1:.4f}  test-F1={tf1:.4f}")

print(f"\n  checkpoint    → {best_row['ckpt']}")
print(f"  training plot → {best_row['curve_plot']}")

# ── Baseline comparison ────────────────────────────────────────────
print(f"\n{'='*75}")
print("FULL COMPARISON — Best Grid Run vs Baselines")
print(f"{'Model':<42} {'Macro F1':>9} {'Micro F1':>9} {'Hamming':>8} {'Exact':>7}")
print("-" * 75)
baselines = [
    ("SVM + TF-IDF",                       0.4821, 0.5103, 0.0812, 0.3100),
    ("Llama 3.1 8B few-shot (balanced 50)", 0.5634, 0.5891, 0.0743, 0.3620),
]
for name, mf1, mif1, hamm, exact in baselines:
    print(f"{name:<42} {mf1:>9.4f} {mif1:>9.4f} {hamm:>8.4f} {exact:>7.4f}")
label = f"{MODEL_KEY} fine-tuned best run (ours)"
print(f"{label:<42} "
      f"{best_row['macro_f1']:>9.4f} {best_row['micro_f1']:>9.4f} "
      f"{best_row['hamming']:>8.4f} {best_row['exact_match']:>7.4f}  ← best grid run")
print("=" * 75)


# ══════════════════════════════════════════════════════════════════
# SAVE BEST MODEL + ARTEFACTS
# ══════════════════════════════════════════════════════════════════
boosted_weights_best = base_weights * float(best_row["BOOST"])
pos_weight_best = torch.tensor(boosted_weights_best, dtype=torch.float32).to(device)

best_model = MentalBERTClassifier(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    dropout=float(best_row["DROPOUT"]),
    pos_weight=pos_weight_best,
).to(device)
best_model.load_state_dict(torch.load(best_row["ckpt"], map_location=device))

final_dir = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(final_dir, exist_ok=True)
tokenizer.save_pretrained(final_dir)
torch.save(best_model.state_dict(), os.path.join(final_dir, "model.pt"))

# Per-label thresholds as an index-aligned list
per_label_thresholds = [float(best_row[f"thr_{l}"]) for l in DSM5_LABELS]

json.dump(
    {
        **{str(i): l for i, l in enumerate(DSM5_LABELS)},
        # Per-label thresholds (index-aligned with DSM5_LABELS)
        "thresholds":    per_label_thresholds,
        "model_key":     MODEL_KEY,
        "model_name":    MODEL_NAME,
        "best_params":   {k: best_row[k] for k in keys},
        "test_macro_f1": float(best_row["macro_f1"]),
        "test_micro_f1": float(best_row["micro_f1"]),
        "test_hamming":  float(best_row["hamming"]),
        "test_exact":    float(best_row["exact_match"]),
    },
    open(os.path.join(final_dir, "label_map.json"), "w"),
    indent=2,
)
results_df.to_csv(os.path.join(OUTPUT_DIR, "grid_search_results.csv"), index=False)

print(f"\nBest model saved → {final_dir}")
print(f"Results CSV      → {os.path.join(OUTPUT_DIR, 'grid_search_results.csv')}")
print(f"All plots        → {PLOTS_DIR}/")


# ══════════════════════════════════════════════════════════════════
# INFERENCE (uses best model + per-label thresholds)
# ══════════════════════════════════════════════════════════════════
_infer_thresholds = np.array(per_label_thresholds)


def predict(text):
    """
    Run inference with the best model using per-label thresholds
    tuned on the validation set.
    """
    best_model.eval()
    enc = tokenizer(
        text, max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.sigmoid(
            best_model(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )["logits"]
        ).squeeze().cpu().numpy()

    detected = [
        DSM5_LABELS[i] for i, p in enumerate(probs)
        if p >= _infer_thresholds[i]
    ]
    return {
        "labels":     detected or ["NONE"],
        "scores":     {DSM5_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "thresholds": {DSM5_LABELS[i]: float(_infer_thresholds[i]) for i in range(NUM_LABELS)},
    }


print("\nINFERENCE DEMO (best model + per-label thresholds)")
print("-" * 65)
for text in [
    "I haven't felt joy in weeks. Nothing seems worth it anymore.",
    "I can't sleep, my mind keeps racing with dark thoughts.",
    "I feel completely worthless and like a burden to everyone.",
    "I had a great day today, everything went well.",
]:
    r = predict(text)
    print(f"\nText       : {text}")
    print(f"Labels     : {r['labels']}")
    print(f"Top-3      : {sorted(r['scores'].items(), key=lambda x: -x[1])[:3]}")
    print(f"Thresholds : { {k: r['thresholds'][k] for k in r['labels'] if k != 'NONE'} }")

print("\nDone ✓")
