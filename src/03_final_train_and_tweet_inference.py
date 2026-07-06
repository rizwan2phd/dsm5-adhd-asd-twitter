# ══════════════════════════════════════════════════════════════════
# DSM-5 MULTI-LABEL CLASSIFIER — FINAL PIPELINE
#
# Stage 1 : Train on ALL labelled data with best params from grid
#           search (script 02) for a fixed number of epochs.
# Stage 2 : Apply the trained classifier to the full tweet corpus and
#           save per-symptom sigmoid scores only (no raw text is
#           retained in the shared output — see data/README.md).
#
# Corresponds to Methods: "Stage 2: Multi-Class Model Training" and
# "User Symptom Scoring and Quality Gating".
#
# Trained on ReDSM5 (redsm5_M5.csv) — cite:
#   Bao E, Pérez A, Parapar J. ReDSM5: a Reddit dataset for DSM-5
#   depression detection. In: Proceedings of the 34th ACM International
#   Conference on Information and Knowledge Management (CIKM '25). 2025.
# ══════════════════════════════════════════════════════════════════

import os, json, warnings, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# BEST HYPERPARAMETERS  (fixed — from grid search, script 02)
# ══════════════════════════════════════════════════════════════════
LR              = 2e-05
DROPOUT         = 0.4
LABEL_SMOOTHING = 0.0
BOOST           = 1.5
EPOCHS          = 9

# ── Other fixed settings ───────────────────────────────────────────
MODEL_KEY    = "mentalroberta"
OUTPUT_DIR   = "./final_outputs"
TWEET_CSV    = "predictions_1m_fullNLI_probs.csv"        # ← set path to your large tweet CSV
TWEET_COL    = "tweet"                     # column name that holds tweet text

MODEL_REGISTRY = {
    "bertbase":      "google-bert/bert-base-uncased",
    "robertaBase":   "FacebookAI/roberta-base",
    "bertweet":      "vinai/bertweet-base",
    "mentalbert":    "mental/mental-bert-base-uncased",
    "mentalroberta": "mental/mental-roberta-base",
    "biov12":        "dmis-lab/biobert-base-cased-v1.2",
    "bioCliBert":    "emilyalsentzer/Bio_ClinicalBERT",
}
MODEL_NAME = MODEL_REGISTRY[MODEL_KEY]

DSM5_LABELS = [
    "DEPRESSED_MOOD", "WORTHLESSNESS", "SUICIDAL_THOUGHTS",
    "FATIGUE", "ANHEDONIA", "SLEEP_ISSUES", "COGNITIVE_ISSUES",
    "APPETITE_CHANGE", "PSYCHOMOTOR",
]
NUM_LABELS   = len(DSM5_LABELS)
MAX_LEN      = 128
BATCH_SIZE   = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED         = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════
print("\nLoading labelled data ...")
df = pd.read_csv("redsm5_M5.csv")
df = df[df["DSM5_symptom"] != "SPECIAL_CASE"]
df = df[df["status"] != 0]
df["DSM5_symptom"] = df["DSM5_symptom"].str.strip().str.upper()

grouped = (
    df.groupby(["sentence_id", "sentence_text"])["DSM5_symptom"]
    .apply(list)
    .reset_index()
    .rename(columns={"DSM5_symptom": "labels"})
)
print(f"Unique sentences : {len(grouped)}")

mlb          = MultiLabelBinarizer(classes=DSM5_LABELS)
label_matrix = mlb.fit_transform(grouped["labels"]).astype(np.float32)
texts        = grouped["sentence_text"].tolist()

# Class weights (inverse frequency × BOOST)
label_counts   = label_matrix.sum(axis=0)
base_weights   = (len(label_matrix) - label_counts) / (label_counts + 1e-6)
pos_weight_arr = base_weights * BOOST

# ══════════════════════════════════════════════════════════════════
# DATASET & MODEL
# ══════════════════════════════════════════════════════════════════
class DSM5Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tok = texts, labels, tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx]),
        }


class MentalBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout, pos_weight):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.pos_weight = pos_weight

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = self.dropout(out.last_hidden_state[:, 0, :].float())
        logits = self.classifier(cls)
        loss   = None
        if labels is not None:
            targets = labels.float()
            if LABEL_SMOOTHING > 0:
                targets = targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits, targets)
        return {"loss": loss, "logits": logits}


# ══════════════════════════════════════════════════════════════════
# TRAIN HELPER
# ══════════════════════════════════════════════════════════════════
def build_optimizer_scheduler(model, n_steps):
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad],
         "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.classifier.parameters(),
         "lr": LR * 10, "weight_decay": WEIGHT_DECAY},
    ], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * WARMUP_RATIO),
        num_training_steps=n_steps,
    )
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler):
    model.train(); total = 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        optimizer.zero_grad()
        out  = model(ids, mask, lbls)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += out["loss"].item()
    return total / len(loader)


# ══════════════════════════════════════════════════════════════════
# STAGE 1 — TRAIN ON ALL LABELLED DATA
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STAGE 1 — Training on ALL labelled data")
print(f"{'='*70}")

print(f"\nLoading tokenizer from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pw    = torch.tensor(pos_weight_arr, dtype=torch.float32).to(device)
model = MentalBERTClassifier(MODEL_NAME, NUM_LABELS, DROPOUT, pw).to(device)

all_loader = DataLoader(
    DSM5Dataset(texts, label_matrix, tokenizer),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
)
optimizer, scheduler = build_optimizer_scheduler(model, len(all_loader) * EPOCHS)

print(f"\nTraining on {len(texts)} sentences for {EPOCHS} epochs ...")
print(f"LR={LR}  DROPOUT={DROPOUT}  BOOST={BOOST}  WARMUP_RATIO={WARMUP_RATIO}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    t0      = time.time()
    tr_loss = train_epoch(model, all_loader, optimizer, scheduler)
    print(f"Ep {epoch:02d}/{EPOCHS} | Loss: {tr_loss:.4f} | {time.time()-t0:.0f}s")

# ── Save checkpoint & config ───────────────────────────────────────
final_ckpt = os.path.join(OUTPUT_DIR, "final_model.pt")
torch.save(model.state_dict(), final_ckpt)
tokenizer.save_pretrained(OUTPUT_DIR)
json.dump(
    {
        **{str(i): l for i, l in enumerate(DSM5_LABELS)},
        "model_key":    MODEL_KEY,
        "model_name":   MODEL_NAME,
        "epochs":       EPOCHS,
        "lr":           LR,
        "dropout":      DROPOUT,
        "boost":        BOOST,
        "warmup_ratio": WARMUP_RATIO,
    },
    open(os.path.join(OUTPUT_DIR, "model_config.json"), "w"),
    indent=2,
)
print(f"\nFinal model saved       → {final_ckpt}")
print(f"Tokenizer saved         → {OUTPUT_DIR}")
print(f"Model config saved      → {os.path.join(OUTPUT_DIR, 'model_config.json')}")

# ══════════════════════════════════════════════════════════════════
# STAGE 2 — APPLY TO LARGE TWEET CSV (sigmoid scores only)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"STAGE 2 — Inference on large tweet file: {TWEET_CSV}")
print(f"{'='*70}")

if not os.path.exists(TWEET_CSV):
    print(f"[WARNING] Tweet CSV not found at '{TWEET_CSV}'. Skipping Stage 2.")
    print("          Set TWEET_CSV at the top of this script and re-run.")
else:
    tweet_df = pd.read_csv(TWEET_CSV)
    if TWEET_COL not in tweet_df.columns:
        raise ValueError(f"Column '{TWEET_COL}' not found in {TWEET_CSV}. "
                         f"Available: {list(tweet_df.columns)}")

    tweet_texts  = tweet_df[TWEET_COL].fillna("").tolist()
    total_tweets = len(tweet_texts)
    print(f"Total tweets to process: {total_tweets:,}")

    # ── Inference dataset (label-free) ───────────────────────────
    class TweetDataset(Dataset):
        def __init__(self, texts, tokenizer):
            self.texts, self.tok = texts, tokenizer

        def __len__(self): return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(
                self.texts[idx], max_length=MAX_LEN,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

    INFER_BATCH  = 128
    tweet_loader = DataLoader(
        TweetDataset(tweet_texts, tokenizer),
        batch_size=INFER_BATCH, shuffle=False, num_workers=4,
    )

    model.eval()
    all_probs = []
    t_start   = time.time()

    with torch.no_grad():
        for batch_i, batch in enumerate(tweet_loader):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            logits = model(ids, mask)["logits"]
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            # Progress report every 5 000 batches
            if (batch_i + 1) % 5000 == 0:
                done    = (batch_i + 1) * INFER_BATCH
                elapsed = time.time() - t_start
                rate    = done / elapsed
                eta     = (total_tweets - done) / rate
                print(f"  Processed {done:>10,} / {total_tweets:,} tweets  "
                      f"| {rate:,.0f} tweets/s  | ETA {eta/60:.1f} min")

    all_probs = np.vstack(all_probs)   # shape: (N, NUM_LABELS)
    print(f"\nInference done in {(time.time()-t_start)/60:.1f} min")

    # ── Write sigmoid scores only ─────────────────────────────────
    score_cols = []
    for i, lbl in enumerate(DSM5_LABELS):
        col = f"score_{lbl}"
        tweet_df[col] = np.round(all_probs[:, i], 4)
        score_cols.append(col)

    out_csv = os.path.join(OUTPUT_DIR, "tweets_annotated_2.csv")
    tweet_df.to_csv(out_csv, index=False)
    print(f"Annotated tweet CSV saved → {out_csv}")
    print(f"Score columns added       : {score_cols}")

# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PIPELINE COMPLETE")
print(f"  Final model   → {final_ckpt}")
if os.path.exists(TWEET_CSV):
    print(f"  Scored tweets → {os.path.join(OUTPUT_DIR, 'tweets_annotated_2.csv')}")
print(f"{'='*70}")
