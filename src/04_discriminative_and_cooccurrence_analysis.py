# ══════════════════════════════════════════════════════════════════
# Group-Level Comparison: Discriminative Analysis + Symptom
# Co-occurrence Analysis
# ══════════════════════════════════════════════════════════════════
# Corresponds to Methods: "Group-Level Comparison" (Per-User Symptom
# Vectors and User-Level Centering, Pre-Filter Zero-Shot Threshold
# Sensitivity Design, L1-Penalised Logistic Regression with Nested
# Cross-Validation, Bootstrap Stability and Robustness Criteria) and
# "Symptom Co-occurrence Analysis".
#
# PREREQUISITE
# ------------
# This script assumes a pandas DataFrame `df` is already loaded in
# memory (e.g. read from the scored-tweets CSV produced by script 03)
# with (at minimum) these columns:
#
#   tweet_id, tweet, user_id, disorder, predicted_prob,
#   score_DEPRESSED_MOOD, score_WORTHLESSNESS, score_SUICIDAL_THOUGHTS,
#   score_FATIGUE, score_ANHEDONIA, score_SLEEP_ISSUES,
#   score_COGNITIVE_ISSUES, score_APPETITE_CHANGE, score_PSYCHOMOTOR
#
# where:
#   - predicted_prob = Stage 1 zero-shot depression-relevance score
#   - score_<SYMPTOM> = Stage 2 per-tweet sigmoid symptom scores
#   - disorder ∈ {'Adhd_eng', 'Asd_eng'}
#
# For example:
#   import pandas as pd
#   df = pd.read_csv("tweets_annotated_2.csv")
#   exec(open("04_discriminative_and_cooccurrence_analysis.py").read())
#
# Per the paper's data-sharing / ethics statement, only tweet_id and
# per-symptom scores (not raw tweet text) are intended for public
# release — see data/README.md.
# ══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

SYMPTOMS = [
    'DEPRESSED_MOOD', 'WORTHLESSNESS', 'SUICIDAL_THOUGHTS', 'FATIGUE',
    'ANHEDONIA', 'SLEEP_ISSUES', 'COGNITIVE_ISSUES', 'APPETITE_CHANGE',
    'PSYCHOMOTOR',
]
SCORE_COLS = [f'score_{s}' for s in SYMPTOMS]
THRESHOLDS  = np.arange(0.45, 0.70, 0.05).round(2)   # [0.45, 0.50, ... 0.65]
N_BOOT      = 1000
rng         = np.random.default_rng(42)

# ── per-tweet, per-symptom score gate ────────────────────────────────────────
# When aggregating tweets into a per-user score vector, only tweets where at
# least one symptom score exceeds that symptom's own threshold are included.
# These are the per-label calibrated thresholds from the Stage 2 classifier
# (script 02 / label_map.json), reproduced here for the score-gate step.
SYMPTOM_THRESHOLDS = {
    'DEPRESSED_MOOD':    0.825,
    'WORTHLESSNESS':     0.575,
    'SUICIDAL_THOUGHTS': 0.150,
    'FATIGUE':           0.425,
    'ANHEDONIA':         0.425,
    'SLEEP_ISSUES':      0.600,
    'COGNITIVE_ISSUES':  0.300,
    'APPETITE_CHANGE':   0.450,
    'PSYCHOMOTOR':       0.450,
}
# Aligned with SCORE_COLS order (score_<SYMPTOM>), used for a vectorized compare.
THRESH_VEC = np.array([SYMPTOM_THRESHOLDS[s] for s in SYMPTOMS])

# ── per-user tweet-count quality gate ────────────────────────────────────────
# After the symptom-score gate, a user is only kept if they still have at
# least this many (gated) tweets. Set to 0 / None-equivalent to disable.
MIN_TWEETS_PER_USER = 30

# ══════════════════════════════════════════════════════════════════
# PART A — DISCRIMINATIVE ANALYSIS (ADHD vs ASD)
# ══════════════════════════════════════════════════════════════════

# ── containers for summary tables ────────────────────────────────────────────
summary_rows = []    # one row per threshold: AUC, accuracy, n_users, n_tweets etc.
coef_rows    = []    # one row per (threshold, symptom): coef + CI + selection

# ---- 0. corpus-level totals (no filtering applied) --------------------------
n_users_total  = df['user_id'].nunique()
n_tweets_total = df['tweet_id'].nunique()
print(f"Full corpus (no quality gate): {n_users_total} users, "
      f"{n_tweets_total} tweets")

for thr in THRESHOLDS:
    print(f"\n{'='*60}")
    print(f"  predicted_prob >= {thr:.2f}")
    print(f"{'='*60}")

    # ---- 1. filter by prediction threshold ---------------------------------
    df1 = df[df['predicted_prob'] >= thr].copy()

    n_users_thr  = df1['user_id'].nunique()
    n_tweets_thr = df1['tweet_id'].nunique()
    print(f"  Users at threshold : {n_users_thr}")
    print(f"  Tweets at threshold: {n_tweets_thr}")

    if df1.empty:
        print(f"  Skipped — no tweets pass threshold {thr:.2f}")
        continue

    # ---- 1b. per-tweet, per-symptom score gate -----------------------------
    # Only keep tweets where at least one symptom score exceeds that
    # symptom's own threshold (from SYMPTOM_THRESHOLDS).
    above_thresh = df1[SCORE_COLS].values > THRESH_VEC   # broadcasts per column
    df1_gated = df1[above_thresh.any(axis=1)].copy()

    n_users_gated  = df1_gated['user_id'].nunique()
    n_tweets_gated = df1_gated['tweet_id'].nunique()
    print(f"  Symptom-score gate  : per-symptom thresholds (see SYMPTOM_THRESHOLDS)")
    print(f"  Users after gate    : {n_users_gated}")
    print(f"  Tweets after gate   : {n_tweets_gated}")

    if df1_gated.empty:
        print(f"  Skipped — no tweets pass symptom-score gate at threshold {thr:.2f}")
        continue

    # ---- 1c. per-user minimum tweet-count floor ----------------------------
    # Keep only users who still have >= MIN_TWEETS_PER_USER tweets after the
    # symptom-score gate above.
    tweet_counts = df1_gated.groupby('user_id')['tweet_id'].nunique()
    users_enough_tweets = tweet_counts[tweet_counts >= MIN_TWEETS_PER_USER].index
    df1_gated = df1_gated[df1_gated['user_id'].isin(users_enough_tweets)].copy()

    n_users_min  = df1_gated['user_id'].nunique()
    n_tweets_min = df1_gated['tweet_id'].nunique()
    print(f"  Min tweets/user gate: >= {MIN_TWEETS_PER_USER}")
    print(f"  Users after min gate: {n_users_min}")
    print(f"  Tweets after min gate: {n_tweets_min}")

    if df1_gated.empty:
        print(f"  Skipped — no users with >= {MIN_TWEETS_PER_USER} tweets at threshold {thr:.2f}")
        continue

    # ---- 2. per-user vector + mean-center ---------------------------------
    user_vectors = (
        df1_gated.groupby('user_id')
           .agg({**{c: 'mean' for c in SCORE_COLS}, 'disorder': 'first'})
           .reset_index()
    )

    X = user_vectors[SCORE_COLS].values
    X = X - X.mean(axis=1, keepdims=True)
    y = (user_vectors['disorder'] == 'Adhd_eng').astype(int).values

    n_users = len(user_vectors)
    n_adhd  = int(y.sum())
    n_asd   = int((1 - y).sum())

    if n_adhd == 0 or n_asd == 0:
        print(f"  Skipped — only one class present at threshold {thr:.2f}")
        continue

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # ---- 3. L1 logistic regression with CV-chosen C -----------------------
    clf = LogisticRegressionCV(
        Cs=20, cv=5, penalty='l1', solver='liblinear',
        class_weight='balanced', scoring='roc_auc',
        max_iter=2000, refit=True,
    )
    clf.fit(X_std, y)

    cv_auc = cross_val_score(
        LogisticRegression(C=clf.C_[0], penalty='l1', solver='liblinear',
                           class_weight='balanced', max_iter=2000),
        X_std, y, cv=5, scoring='roc_auc'
    )

    print(f"  Users: {n_users}  | ADHD: {n_adhd}  ASD: {n_asd}")
    print(f"  Tweets used (post gates + per-symptom aggregation): {n_tweets_min}")
    print(f"  Chosen C : {clf.C_[0]:.4f}")
    print(f"  Train acc: {clf.score(X_std, y):.3f}")
    print(f"  CV AUC   : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # ---- 4. bootstrap -----------------------------------------------------
    n          = X_std.shape[0]
    boot_coefs = np.full((N_BOOT, len(SYMPTOMS)), np.nan)

    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb  = y[idx]
        if yb.min() == yb.max():
            continue
        m = LogisticRegression(C=clf.C_[0], penalty='l1', solver='liblinear',
                               class_weight='balanced', max_iter=2000)
        m.fit(X_std[idx], yb)
        boot_coefs[b] = m.coef_[0]

    ci_low     = np.nanpercentile(boot_coefs,  2.5, axis=0)
    ci_high    = np.nanpercentile(boot_coefs, 97.5, axis=0)
    select_freq = np.mean(boot_coefs != 0, axis=0)
    mean_co    = np.nanmean(boot_coefs, axis=0)
    stable     = (ci_low > 0) | (ci_high < 0)

    # ---- 5. per-threshold coefficient table -------------------------------
    coef_df = (
        pd.DataFrame({
            'symptom':     SYMPTOMS,
            'coef':        clf.coef_[0],
            'boot_mean':   mean_co,
            'ci_low':      ci_low,
            'ci_high':     ci_high,
            'select_freq': select_freq,
            'stable_95':   stable,
            'favors':      np.where(clf.coef_[0] > 0, 'ADHD',
                           np.where(clf.coef_[0] < 0, 'ASD', '—')),
        })
        .assign(abs_coef=lambda d: d['coef'].abs())
        .sort_values('abs_coef', ascending=False)
        .drop(columns='abs_coef')
        .reset_index(drop=True)
    )
    print(coef_df.to_string(index=False))

    # ---- 6. store for cross-threshold summary -----------------------------
    summary_rows.append({
        'threshold':    thr,
        'n_users':      n_users,
        'n_tweets':     n_tweets_min,
        'n_adhd':       n_adhd,
        'n_asd':        n_asd,
        'chosen_C':     clf.C_[0],
        'train_acc':    clf.score(X_std, y),
        'cv_auc_mean':  cv_auc.mean(),
        'cv_auc_std':   cv_auc.std(),
    })
    for _, row in coef_df.iterrows():
        coef_rows.append({'threshold': thr, **row.to_dict()})

# ── cross-threshold summary tables ───────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
coef_all   = pd.DataFrame(coef_rows)

print("\n\n" + "="*60)
print("  CROSS-THRESHOLD PERFORMANCE SUMMARY (per-symptom score gate + min-tweets gate applied)")
print("="*60)
print(summary_df.to_string(index=False))

print("\n\n" + "="*60)
print("  CROSS-THRESHOLD SELECTION FREQUENCY PER SYMPTOM")
print("="*60)
pivot_sel = (
    coef_all.pivot_table(
        index='symptom', columns='threshold',
        values='select_freq', aggfunc='first'
    )
    .round(3)
)
# sort by mean selection frequency across thresholds
pivot_sel['mean'] = pivot_sel.mean(axis=1)
pivot_sel = pivot_sel.sort_values('mean', ascending=False).drop(columns='mean')
print(pivot_sel.to_string())

print("\n\n" + "="*60)
print("  CROSS-THRESHOLD COEFFICIENT DIRECTION PER SYMPTOM")
print("="*60)
pivot_dir = (
    coef_all.pivot_table(
        index='symptom', columns='threshold',
        values='favors', aggfunc='first'
    )
)
# sort same order as selection pivot
pivot_dir = pivot_dir.loc[pivot_sel.index]
print(pivot_dir.to_string())


# ══════════════════════════════════════════════════════════════════
# PART B — SYMPTOM CO-OCCURRENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr

SHORT_NAMES = {
    'DEPRESSED_MOOD':    'Dep.Mood',
    'WORTHLESSNESS':     'Worthless',
    'SUICIDAL_THOUGHTS': 'Suicidal',
    'FATIGUE':           'Fatigue',
    'ANHEDONIA':         'Anhedonia',
    'SLEEP_ISSUES':      'Sleep',
    'COGNITIVE_ISSUES':  'Cognitive',
    'APPETITE_CHANGE':   'Appetite',
    'PSYCHOMOTOR':       'Psychomotor',
}
SHORT_COLS  = list(SHORT_NAMES.values())

N_BOOT   = 1000     # bootstrap replicates per (threshold, group, pair)
CI_ALPHA = 0.05      # 95% CI
RNG      = np.random.default_rng(42)   # single seeded generator, reused everywhere

# NOTE: MIN_TWEETS_PER_USER, SYMPTOM_THRESHOLDS, THRESH_VEC, THRESHOLDS,
# SYMPTOMS, and SCORE_COLS are all reused from Part A above so that the
# co-occurrence analysis operates on exactly the same gated per-user
# profiles as the discriminative analysis.


# ── helper: vectorized bootstrap of Pearson r for one pair ───────────────────
def bootstrap_pearson(x, y, n_boot=N_BOOT, rng=RNG):
    """
    Bootstrap the Pearson correlation between x and y by resampling
    observations (rows / users) with replacement.

    Returns the array of n_boot bootstrap replicate correlations.
    Fully vectorized: builds an (n_boot, n) index matrix instead of
    looping n_boot times calling scipy.stats.pearsonr.
    """
    n = len(x)
    if n < 3:
        return np.full(n_boot, np.nan)

    idx = rng.integers(0, n, size=(n_boot, n))   # (n_boot, n) resample indices
    xb = x[idx]
    yb = y[idx]

    xb = xb - xb.mean(axis=1, keepdims=True)
    yb = yb - yb.mean(axis=1, keepdims=True)

    num = (xb * yb).sum(axis=1)
    den = np.sqrt((xb ** 2).sum(axis=1) * (yb ** 2).sum(axis=1))

    with np.errstate(invalid='ignore', divide='ignore'):
        r = num / den
    return r


def percentile_ci(boot_samples, alpha=CI_ALPHA):
    lo, hi = np.nanpercentile(boot_samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ── 1. Collect correlations + bootstrap CIs for every threshold ──────────────
# all_corrs[group][pair]      -> list of point-estimate r, one per threshold
# all_boot[group][pair][thr]  -> array of n_boot bootstrap replicate r's
all_corrs = {'ADHD': {}, 'ASD': {}}
all_boot  = {'ADHD': {}, 'ASD': {}}
thr_meta  = []

for thr in THRESHOLDS:
    # ---- step 1: predicted_prob (zero-shot) threshold -----------------------
    df1 = df[df['predicted_prob'] >= thr].copy()
    n_users_at_threshold  = df1['user_id'].nunique()
    n_tweets_at_threshold = len(df1)

    # ---- step 2: per-tweet symptom-relevance gate ----------------------------
    # Keep only tweets where at least one symptom score exceeds that
    # symptom's own calibrated threshold (SYMPTOM_THRESHOLDS).
    above_thresh = df1[SCORE_COLS].values > THRESH_VEC   # broadcasts per column
    df1 = df1[above_thresh.any(axis=1)].copy()
    n_users_after_gate  = df1['user_id'].nunique()
    n_tweets_after_gate = len(df1)

    # ---- step 3: per-user minimum tweet-count floor (on gated tweets) -------
    tweet_counts   = df1.groupby('user_id').size().rename('n_tweets')
    n_users_before = tweet_counts.shape[0]
    eligible_users = tweet_counts[tweet_counts >= MIN_TWEETS_PER_USER].index
    n_users_after  = len(eligible_users)

    df1 = df1[df1['user_id'].isin(eligible_users)].copy()

    print(f"threshold={thr:.2f}  at-threshold: {n_users_at_threshold} users / "
          f"{n_tweets_at_threshold} tweets  ->  after symptom gate: "
          f"{n_users_after_gate} users / {n_tweets_after_gate} tweets  ->  "
          f"after min-tweets gate: {n_users_after} users / {len(df1)} tweets")

    if df1.empty:
        print(f"  Skipped — no tweets pass all gates at threshold {thr:.2f}")
        continue

    user_vectors = (
        df1.groupby('user_id')
           .agg({**{c: 'mean' for c in SCORE_COLS}, 'disorder': 'first'})
           .reset_index()
    )
    # mean-center per user (removes overall severity/response-style effect)
    X = user_vectors[SCORE_COLS].values
    X = X - X.mean(axis=1, keepdims=True)
    user_vectors[SCORE_COLS] = X

    adhd_df = user_vectors[user_vectors['disorder'] == 'Adhd_eng'][SCORE_COLS].copy()
    asd_df  = user_vectors[user_vectors['disorder'] == 'Asd_eng' ][SCORE_COLS].copy()
    adhd_df.columns = SHORT_COLS
    asd_df.columns  = SHORT_COLS

    thr_meta.append({
        'threshold':                   thr,
        'n_users_at_threshold':        n_users_at_threshold,
        'n_tweets_at_threshold':       n_tweets_at_threshold,
        'n_users_after_symptom_gate':  n_users_after_gate,
        'n_tweets_after_symptom_gate': n_tweets_after_gate,
        'n_users_after_tweet_filter':  n_users_after,
        'n_tweets_final':              len(df1),
        'n_adhd':                      len(adhd_df),
        'n_asd':                       len(asd_df),
    })

    for group, gdf in [('ADHD', adhd_df), ('ASD', asd_df)]:
        for i in range(len(SHORT_COLS)):
            for j in range(i + 1, len(SHORT_COLS)):
                si, sj = SHORT_COLS[i], SHORT_COLS[j]
                pair   = f'{si} × {sj}'

                x = gdf[si].values
                y = gdf[sj].values

                r, _ = pearsonr(x, y)
                all_corrs[group].setdefault(pair, []).append(r)

                boot_r = bootstrap_pearson(x, y)
                all_boot[group].setdefault(pair, {})[thr] = boot_r

thr_meta_df = pd.DataFrame(thr_meta)
print()
print(thr_meta_df.to_string(index=False))


# ── 2. Build stability summary dataframe (now with bootstrap CIs) ────────────
rows = []
for group in ['ADHD', 'ASD']:
    for pair, rs in all_corrs[group].items():
        rs_arr = np.array(rs)

        # per-threshold bootstrap CIs for this pair
        ci_los, ci_his = [], []
        for thr in THRESHOLDS:
            if thr not in all_boot[group][pair]:
                ci_los.append(np.nan)
                ci_his.append(np.nan)
                continue
            lo, hi = percentile_ci(all_boot[group][pair][thr])
            ci_los.append(lo)
            ci_his.append(hi)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)

        # CI excludes zero at a given threshold if lo and hi share sign
        ci_excludes_zero = (ci_los > 0) | (ci_his < 0)

        rows.append({
            'group':                 group,
            'pair':                  pair,
            'mean_r':                rs_arr.mean(),
            'std_r':                 rs_arr.std(),
            'min_r':                 rs_arr.min(),
            'max_r':                 rs_arr.max(),
            'range_r':               rs_arr.max() - rs_arr.min(),
            'sign_stable':           bool(np.all(rs_arr > 0) | np.all(rs_arr < 0)),
            'ci_lo':                 ci_los,          # per-threshold arrays
            'ci_hi':                 ci_his,
            'ci_excludes_zero_all':  bool(np.all(ci_excludes_zero)),
            'ci_excludes_zero_any':  bool(np.any(ci_excludes_zero)),
            # "robust": same sign everywhere AND never plausibly zero
            'robust_stable':         bool((np.all(rs_arr > 0) | np.all(rs_arr < 0))
                                           and np.all(ci_excludes_zero)),
            'rs':                    rs_arr,
        })

stab_df = pd.DataFrame(rows)


# ── 3. Print stability summary ────────────────────────────────────────────────
for group in ['ADHD', 'ASD']:
    g = stab_df[stab_df['group'] == group].copy()
    print(f"\n{'='*75}")
    print(f"  {group} — Correlation Stability Across Thresholds "
          f"({', '.join(str(t) for t in THRESHOLDS)})")
    print(f"{'='*75}")
    print(
        g[['pair', 'mean_r', 'std_r', 'range_r',
           'sign_stable', 'ci_excludes_zero_all', 'robust_stable']]
          .sort_values('mean_r')
          .to_string(index=False)
    )


# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 24))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

# ─── 4a & 4b: mean correlation heatmap per group ─────────────────────────────
for col_idx, group in enumerate(['ADHD', 'ASD']):
    mean_mat = pd.DataFrame(np.eye(len(SHORT_COLS)), columns=SHORT_COLS, index=SHORT_COLS)
    for i in range(len(SHORT_COLS)):
        for j in range(i + 1, len(SHORT_COLS)):
            si, sj = SHORT_COLS[i], SHORT_COLS[j]
            pair   = f'{si} × {sj}'
            r_mean = np.mean(all_corrs[group][pair])
            mean_mat.loc[si, sj] = r_mean
            mean_mat.loc[sj, si] = r_mean

    ax = fig.add_subplot(gs[0, col_idx])
    sns.heatmap(
        mean_mat, ax=ax, annot=mean_mat.round(2), fmt='.2f',
        cmap='coolwarm', vmin=-1, vmax=1, center=0,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8, 'label': 'Mean Pearson r'},
    )
    ax.set_title(f'{group} — Mean Correlation Across All Thresholds',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# ─── 4c & 4d: stability heatmap — fraction of thresholds where CI excludes 0
for col_idx, group in enumerate(['ADHD', 'ASD']):
    frac_mat = pd.DataFrame(np.zeros((len(SHORT_COLS), len(SHORT_COLS))),
                            columns=SHORT_COLS, index=SHORT_COLS)
    for i in range(len(SHORT_COLS)):
        for j in range(i + 1, len(SHORT_COLS)):
            si, sj = SHORT_COLS[i], SHORT_COLS[j]
            pair   = f'{si} × {sj}'
            row = stab_df[(stab_df['group'] == group) & (stab_df['pair'] == pair)].iloc[0]
            ci_lo, ci_hi = row['ci_lo'], row['ci_hi']
            frac_excl_zero = np.nanmean((ci_lo > 0) | (ci_hi < 0))
            frac_mat.loc[si, sj] = frac_excl_zero
            frac_mat.loc[sj, si] = frac_excl_zero

    ax = fig.add_subplot(gs[1, col_idx])
    sns.heatmap(
        frac_mat, ax=ax, annot=frac_mat.round(2), fmt='.2f',
        cmap='YlGnBu', vmin=0, vmax=1,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8, 'label': 'Fraction of thresholds\nwith 95% CI excluding 0'},
    )
    ax.set_title(f'{group} — Statistical Robustness\n'
                 f'(1.0 = CI never crosses zero, any threshold)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# ─── 4e/4f: line plot with bootstrap CI ribbons for top pairs per group ──────
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

for ax, group in [(ax5, 'ADHD'), (ax6, 'ASD')]:
    g = stab_df[stab_df['group'] == group].copy()
    top6 = (g.assign(abs_mean=lambda d: d['mean_r'].abs())
              .sort_values(['robust_stable', 'abs_mean'], ascending=[False, False])
              .head(6))

    colors = plt.cm.tab10(np.linspace(0, 0.9, 6))
    for (_, row), color in zip(top6.iterrows(), colors):
        ax.plot(THRESHOLDS, row['rs'], marker='o', markersize=5, linewidth=2,
                label=row['pair'], color=color)
        ax.fill_between(THRESHOLDS, row['ci_lo'], row['ci_hi'],
                         color=color, alpha=0.15)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(THRESHOLDS)
    ax.set_xlabel('predicted_prob threshold', fontsize=10)
    ax.set_ylabel('Pearson r (with 95% bootstrap CI)', fontsize=10)
    ax.set_title(f'{group} — Top 6 Pairs Across Thresholds\n'
                 f'(ranked by robust stability then |mean r|)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Symptom Co-occurrence Stability Across Predicted Probability Thresholds\n'
             '(gated tweets; with bootstrap confidence intervals)',
             fontsize=15, fontweight='bold', y=0.99)
plt.savefig('symptom_cooccurrence_stability_bootstrap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → symptom_cooccurrence_stability_bootstrap.png")


# ── 5. Final clean summary: robust pairs only ─────────────────────────────────
print("\n\n── ROBUST PAIRS (same sign at every threshold AND 95% CI excludes 0 everywhere) ──")
robust = stab_df[stab_df['robust_stable']].copy()
for group in ['ADHD', 'ASD']:
    g = robust[robust['group'] == group].sort_values('mean_r')
    print(f"\n  {group} ({len(g)} robust pairs):")
    print(g[['pair', 'mean_r', 'std_r', 'range_r']].to_string(index=False))

print("\n\n── SIGN-STABLE BUT NOT STATISTICALLY ROBUST (CI touches 0 at some threshold) ──")
shaky = stab_df[stab_df['sign_stable'] & ~stab_df['robust_stable']].copy()
for group in ['ADHD', 'ASD']:
    g = shaky[shaky['group'] == group].sort_values('mean_r')
    print(f"\n  {group} ({len(g)} pairs):")
    print(g[['pair', 'mean_r', 'std_r', 'range_r']].to_string(index=False))

print("\n\n── UNSTABLE PAIRS (sign flips across thresholds) ──")
unstable = stab_df[~stab_df['sign_stable']].copy()
for group in ['ADHD', 'ASD']:
    g = unstable[unstable['group'] == group].sort_values('range_r', ascending=False)
    print(f"\n  {group} ({len(g)} unstable pairs):")
    print(g[['pair', 'mean_r', 'std_r', 'min_r', 'max_r', 'range_r']].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# ── 6. ADHD vs ASD — which pairs differ, which are shared ───────────────────
# ══════════════════════════════════════════════════════════════════════════
# Reuses all_corrs / all_boot from section 1, so no re-fitting is needed.
# Since ADHD and ASD bootstrap resamples are independent, subtracting the
# two bootstrap arrays element-wise gives a valid empirical distribution of
# the *difference* in r — this is the standard trick for comparing two
# independent bootstrap estimates without a fresh bootstrap.

diff_rows = []
all_pairs = list(all_corrs['ADHD'].keys())   # same pair set for both groups

for pair in all_pairs:
    adhd_r = np.array(all_corrs['ADHD'][pair])
    asd_r  = np.array(all_corrs['ASD'][pair])
    diff_r = adhd_r - asd_r                  # one value per threshold

    diff_ci_lo, diff_ci_hi = [], []
    for thr in THRESHOLDS:
        if thr not in all_boot['ADHD'][pair] or thr not in all_boot['ASD'][pair]:
            diff_ci_lo.append(np.nan)
            diff_ci_hi.append(np.nan)
            continue
        diff_boot = all_boot['ADHD'][pair][thr] - all_boot['ASD'][pair][thr]
        lo, hi = percentile_ci(diff_boot)
        diff_ci_lo.append(lo)
        diff_ci_hi.append(hi)
    diff_ci_lo = np.array(diff_ci_lo)
    diff_ci_hi = np.array(diff_ci_hi)

    diff_excludes_zero   = (diff_ci_lo > 0) | (diff_ci_hi < 0)
    diff_sign_stable     = bool(np.all(diff_r > 0) | np.all(diff_r < 0))
    diff_robust_different = bool(diff_sign_stable and np.all(diff_excludes_zero))

    adhd_row = stab_df[(stab_df['group'] == 'ADHD') & (stab_df['pair'] == pair)].iloc[0]
    asd_row  = stab_df[(stab_df['group'] == 'ASD')  & (stab_df['pair'] == pair)].iloc[0]

    both_robust_same_direction = (
        adhd_row['robust_stable'] and asd_row['robust_stable']
        and np.sign(adhd_row['mean_r']) == np.sign(asd_row['mean_r'])
    )

    if diff_robust_different:
        category = 'Disorder-specific'
    elif both_robust_same_direction and not np.any(diff_excludes_zero):
        category = 'Shared'
    else:
        category = 'Ambiguous'   # underpowered / inconsistent evidence either way

    diff_rows.append({
        'pair':               pair,
        'adhd_mean_r':        adhd_row['mean_r'],
        'asd_mean_r':         asd_row['mean_r'],
        'mean_diff':          diff_r.mean(),
        'std_diff':           diff_r.std(),
        'diff_sign_stable':   diff_sign_stable,
        'diff_robust_different': diff_robust_different,
        'adhd_robust':        adhd_row['robust_stable'],
        'asd_robust':         asd_row['robust_stable'],
        'category':           category,
        'diff_r':             diff_r,
        'diff_ci_lo':         diff_ci_lo,
        'diff_ci_hi':         diff_ci_hi,
    })

diff_df = pd.DataFrame(diff_rows)

print(f"\n\n{'='*75}")
print("  ADHD vs ASD — Differential Symptom Co-occurrence")
print(f"{'='*75}")

print("\n── DISORDER-SPECIFIC (difference sign-stable AND 95% CI excludes 0, "
      "every threshold) ──")
spec = diff_df[diff_df['category'] == 'Disorder-specific'].sort_values(
    'mean_diff', key=lambda s: s.abs(), ascending=False)
print(spec[['pair', 'adhd_mean_r', 'asd_mean_r', 'mean_diff', 'std_diff']]
      .to_string(index=False))

print("\n── SHARED (both groups individually robust, same direction, "
      "difference CI includes 0 everywhere) ──")
shared = diff_df[diff_df['category'] == 'Shared'].sort_values(
    'adhd_mean_r', key=lambda s: s.abs(), ascending=False)
print(shared[['pair', 'adhd_mean_r', 'asd_mean_r', 'mean_diff']]
      .to_string(index=False))

print(f"\n── AMBIGUOUS ({len(diff_df[diff_df['category']=='Ambiguous'])} pairs — "
      "weak/inconsistent evidence for either 'shared' or 'different') ──")
amb = diff_df[diff_df['category'] == 'Ambiguous'].sort_values(
    'mean_diff', key=lambda s: s.abs(), ascending=False)
print(amb[['pair', 'adhd_mean_r', 'asd_mean_r', 'mean_diff', 'adhd_robust', 'asd_robust']]
      .to_string(index=False))


# ── 7. Plot: difference heatmap + top differential pairs bar chart ──────────
fig2 = plt.figure(figsize=(20, 9))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4)

# 7a: symmetric heatmap of ADHD_r - ASD_r, with '*' on robustly-different cells
diff_mat = pd.DataFrame(np.zeros((len(SHORT_COLS), len(SHORT_COLS))),
                        columns=SHORT_COLS, index=SHORT_COLS)
sig_mat = pd.DataFrame(np.full((len(SHORT_COLS), len(SHORT_COLS)), '', dtype=object),
                       columns=SHORT_COLS, index=SHORT_COLS)

for _, row in diff_df.iterrows():
    si, sj = row['pair'].split(' × ')
    diff_mat.loc[si, sj] = row['mean_diff']
    diff_mat.loc[sj, si] = row['mean_diff']
    mark = '*' if row['diff_robust_different'] else ''
    sig_mat.loc[si, sj] = mark
    sig_mat.loc[sj, si] = mark

ax7 = fig2.add_subplot(gs2[0, 0])
annot_labels = diff_mat.round(2).astype(str) + sig_mat
sns.heatmap(
    diff_mat, ax=ax7, annot=annot_labels.values, fmt='', cmap='PuOr',
    vmin=-0.6, vmax=0.6, center=0,
    linewidths=0.5, linecolor='white', annot_kws={'size': 7.5},
    cbar_kws={'shrink': 0.8, 'label': 'mean r (ADHD) − mean r (ASD)'},
)
ax7.set_title('ADHD − ASD Correlation Difference\n'
              '(* = robustly different: sign-stable & CI excludes 0 at every threshold)',
              fontsize=12, fontweight='bold', pad=10)
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=40, ha='right', fontsize=8)
ax7.set_yticklabels(ax7.get_yticklabels(), rotation=0, fontsize=8)

# 7b: top 10 differential pairs, bar chart with bootstrap CI (median threshold as reference)
ax8 = fig2.add_subplot(gs2[0, 1])
ref_idx = len(THRESHOLDS) // 2   # middle threshold used as a representative CI

top10 = diff_df.assign(abs_diff=lambda d: d['mean_diff'].abs()) \
               .sort_values('abs_diff', ascending=False).head(10)

y_pos = np.arange(len(top10))
diffs_ref = top10['diff_r'].apply(lambda a: a[ref_idx]).values
err_lo = diffs_ref - top10['diff_ci_lo'].apply(lambda a: a[ref_idx]).values
err_hi = top10['diff_ci_hi'].apply(lambda a: a[ref_idx]).values - diffs_ref
colors = ['#d95f02' if r > 0 else '#7570b3' for r in diffs_ref]

ax8.barh(y_pos, diffs_ref, xerr=[err_lo, err_hi], color=colors, alpha=0.85,
         capsize=4, error_kw={'linewidth': 1.2})
ax8.set_yticks(y_pos)
ax8.set_yticklabels(top10['pair'], fontsize=9)
ax8.invert_yaxis()
ax8.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax8.set_xlabel(f'ADHD r − ASD r  (at threshold={THRESHOLDS[ref_idx]}, 95% bootstrap CI)',
               fontsize=9)
ax8.set_title('Top 10 Most Differential Pairs\n(orange = stronger in ADHD, purple = stronger in ASD)',
              fontsize=12, fontweight='bold')
ax8.grid(True, axis='x', alpha=0.3)

fig2.suptitle('ADHD vs ASD — Where Symptom Co-occurrence Diverges (gated tweets)',
              fontsize=15, fontweight='bold', y=1.02)
plt.savefig('adhd_asd_differential_cooccurrence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved → adhd_asd_differential_cooccurrence.png")
