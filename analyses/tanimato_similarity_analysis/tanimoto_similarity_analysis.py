# %%
"""
Applicability domain analysis for the sweet/non-sweet classification model.

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── parameters ───────────────────────────────────────────────────────────────

K = 5   

# ── data ─────────────────────────────────────────────────────────────────────

train = pd.read_csv("sweet_nonsweet_train_set.csv")
test  = pd.read_csv("sweet_nonsweet_test_set.csv")

ecfp = [c for c in train.columns if c.startswith("ECFP4_Bit_")]
X_tr = train[ecfp].values.astype(np.float32)
X_te = test[ecfp].values.astype(np.float32)

print(f"Train: {len(train)}  (sweet {int(train['Class'].sum())}, "
      f"non-sweet {int((train['Class']==0).sum())})")
print(f"Test : {len(test)}  (sweet {int(test['Class'].sum())}, "
      f"non-sweet {int((test['Class']==0).sum())})")

# ── Tanimoto ─────────────────────────────────────────────────────────────────

def tanimoto(A, B):
    ab = A @ B.T
    sa = A.sum(1, keepdims=True)
    sb = B.sum(1, keepdims=True)
    d  = sa + sb.T - ab
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, 0.0, ab / d)

# ── training-set k-NN similarity ────────────────────────────────────────────

S_tr = tanimoto(X_tr, X_tr)
np.fill_diagonal(S_tr, -1.0)
knn_tr  = np.sort(S_tr, axis=1)[:, -K:]
msim_tr = knn_tr.mean(axis=1)

# threshold: 10th percentile (non-parametric)
thr = float(np.percentile(msim_tr, 10))

print(f"\nTraining k-NN similarity  mean={msim_tr.mean():.4f}  "
      f"median={np.median(msim_tr):.4f}  sd={msim_tr.std():.4f}")
print(f"AD threshold (10th percentile) = {thr:.4f}")

n_out_tr = int((msim_tr < thr).sum())
print(f"Training compounds below threshold: {n_out_tr}/{len(train)}")

# ── test-set assessment ──────────────────────────────────────────────────────

S_te = tanimoto(X_te, X_tr)
knn_te  = np.sort(S_te, axis=1)[:, -K:]
msim_te = knn_te.mean(axis=1)
max_te  = S_te.max(axis=1)

ad = msim_te >= thr
cl = test["Class"].values

print(f"\nTest AD coverage: {ad.sum()}/{len(test)} "
      f"({100*ad.mean():.1f}%) within AD")
for lab, c in [("Sweet", 1), ("Non-sweet", 0)]:
    m = cl == c
    print(f"  {lab:10s}: {ad[m].sum()}/{m.sum()} within AD")

# ── output CSV ───────────────────────────────────────────────────────────────

out = pd.DataFrame({
    "Name":           test["Name"].values,
    "Class":          cl,
    "Class_label":    np.where(cl == 1, "Sweet", "Non-sweet"),
    "Mean_kNN_Tc":    np.round(msim_te, 4),
    "Max_Tc_to_train": np.round(max_te, 4),
    "Within_AD":      ad,
})
out.sort_values("Mean_kNN_Tc").to_csv("ad_classification_results.csv", index=False)
print("Saved ad_classification_results.csv")

# ── figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.subplots_adjust(hspace=0.38, wspace=0.30)

CI = "#3A7D44"
CO = "#B8342A"
CT = "#1B2631"

# (a) training distribution
ax = axes[0, 0]
ax.hist(msim_tr, bins=30, color="#5B8DB8", edgecolor="white", lw=0.4)
ax.axvline(thr, color=CT, ls="--", lw=1.6,
           label=f"10th percentile = {thr:.3f}")
ax.set_xlabel("Mean Tanimoto similarity (k=5 NN)")
ax.set_ylabel("Count")
ax.set_title("(a) Training set", fontweight="bold", fontsize=10)
ax.legend(fontsize=8.5)

# (b) test distribution
ax = axes[0, 1]
edges = np.histogram_bin_edges(msim_te, bins=25)
ax.hist(msim_te[ad],  bins=edges, color=CI, alpha=0.85,
        label=f"Within AD (n={ad.sum()})", edgecolor="white", lw=0.4)
ax.hist(msim_te[~ad], bins=edges, color=CO, alpha=0.85,
        label=f"Outside AD (n={(~ad).sum()})", edgecolor="white", lw=0.4)
ax.axvline(thr, color=CT, ls="--", lw=1.6)
ax.set_xlabel("Mean Tanimoto similarity (k=5 NN)")
ax.set_ylabel("Count")
ax.set_title("(b) Test set", fontweight="bold", fontsize=10)
ax.legend(fontsize=8.5)

# (c) class breakdown
ax = axes[1, 0]
cats = ["Non-sweet", "Sweet"]
wi = [ad[cl == c].sum() for c in [0, 1]]
oi = [(~ad)[cl == c].sum() for c in [0, 1]]
x  = np.arange(2)
bw = 0.32
b1 = ax.bar(x - bw/2, wi, bw, color=CI, edgecolor="white", label="Within AD")
b2 = ax.bar(x + bw/2, oi, bw, color=CO, edgecolor="white", label="Outside AD")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Compounds")
ax.set_title("(c) Coverage by class", fontweight="bold", fontsize=10)
ax.legend(fontsize=8.5)
for bars in [b1, b2]:
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width()/2, h + 0.4,
                    str(int(h)), ha="center", va="bottom", fontsize=9)

# (d) sorted per-compound
ax = axes[1, 1]
idx = np.argsort(msim_te)
ss  = msim_te[idx]
cc  = np.where(ad[idx], CI, CO)
ax.scatter(np.arange(len(ss)), ss, c=cc, s=14, alpha=0.7, zorder=2)
ax.axhline(thr, color=CT, ls="--", lw=1.6)
ax.set_xlabel("Compounds (sorted)")
ax.set_ylabel("Mean Tanimoto similarity (k=5 NN)")
ax.set_title("(d) Per-compound similarity", fontweight="bold", fontsize=10)
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CI, ms=6,
           label="Within AD"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CO, ms=6,
           label="Outside AD"),
    Line2D([0],[0], color=CT, ls="--", lw=1.6,
           label=f"Threshold = {thr:.3f}"),
], fontsize=8)

fig.suptitle(
    "Applicability domain — sweet/non-sweet classification\n"
    f"ECFP4 Tanimoto k-NN (k={K}), "
    f"threshold = {thr:.4f} (10th percentile)",
    fontsize=11, fontweight="bold", y=0.99)

plt.savefig("ad_classification.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved ad_classification.png")


# %%
"""
Applicability domain analysis for sweetness intensity regression models.

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

# ── data ─────────────────────────────────────────────────────────────────────

df = pd.read_csv("sweeteners_with_molecular_descriptors_and_fingerprints.csv")
ecfp = [c for c in df.columns if c.startswith("ECFP4_Bit_")]
fp   = df[ecfp].values.astype(np.float32)
sw   = df["Sweetness"].values  # logSw

print(f"Dataset: {len(df)} sweeteners, {len(ecfp)}-bit ECFP4")

# ── Tanimoto ─────────────────────────────────────────────────────────────────

K = 5

def tanimoto(A, B):
    ab = A @ B.T
    sa = A.sum(1, keepdims=True)
    sb = B.sum(1, keepdims=True)
    d  = sa + sb.T - ab
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, 0.0, ab / d)


def run_ad(fp, idx_all, rs, model_name):
    """Compute AD for a single model split."""
    idx_tr, idx_te = train_test_split(idx_all, test_size=0.3, random_state=rs)

    # training k-NN similarity
    S_tr = tanimoto(fp[idx_tr], fp[idx_tr])
    np.fill_diagonal(S_tr, -1.0)
    msim_tr = np.sort(S_tr, axis=1)[:, -K:].mean(axis=1)
    thr = float(np.percentile(msim_tr, 10))

    # test k-NN similarity
    S_te    = tanimoto(fp[idx_te], fp[idx_tr])
    msim_te = np.sort(S_te, axis=1)[:, -K:].mean(axis=1)
    max_te  = S_te.max(axis=1)
    ad      = msim_te >= thr

    print(f"\n  {model_name}  (random_state={rs})")
    print(f"  Train: {len(idx_tr)}   Test: {len(idx_te)}")
    print(f"  Training k-NN sim: mean={msim_tr.mean():.4f}  "
          f"median={np.median(msim_tr):.4f}  sd={msim_tr.std():.4f}")
    print(f"  Threshold (10th percentile) = {thr:.4f}")
    print(f"  Test AD coverage: {ad.sum()}/{len(idx_te)} ({100*ad.mean():.1f}%)")

    return idx_tr, idx_te, msim_tr, msim_te, max_te, thr, ad


idx_all = np.arange(len(df))

# Model 1: Voting Ensemble (rs=42)
itr1, ite1, mstr1, mste1, maxe1, thr1, ad1 = run_ad(
    fp, idx_all, 42, "Voting Ensemble regression")

# Model 2: XGBoost (rs=5)
itr2, ite2, mstr2, mste2, maxe2, thr2, ad2 = run_ad(
    fp, idx_all, 5, "XGBoost regression")

# ── CSV outputs ──────────────────────────────────────────────────────────────

for tag, ite, mste, maxe, thr, ad in [
    ("voting_ensemble", ite1, mste1, maxe1, thr1, ad1),
    ("xgboost",         ite2, mste2, maxe2, thr2, ad2),
]:
    out = pd.DataFrame({
        "Molecule":        df["Molecule"].values[ite],
        "logSw":           sw[ite],
        "Mean_kNN_Tc":     np.round(mste, 4),
        "Max_Tc_to_train": np.round(maxe, 4),
        "Within_AD":       ad,
    })
    fname = f"ad_regression_{tag}.csv"
    out.sort_values("Mean_kNN_Tc").to_csv(fname, index=False)
    print(f"Saved {fname}")

# ── figure ───────────────────────────────────────────────────────────────────

CI = "#3A7D44"
CO = "#B8342A"
CT = "#1B2631"

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
fig.subplots_adjust(hspace=0.42, wspace=0.32)

models = [
    ("Voting Ensemble (random_state=42)", mstr1, mste1, thr1, ad1, ite1),
    ("XGBoost (random_state=5)",          mstr2, mste2, thr2, ad2, ite2),
]

labels = [["(a)","(b)","(c)"], ["(d)","(e)","(f)"]]

for row, (title, ms_tr, ms_te, thr, ad, ite) in enumerate(models):

    # training distribution
    ax = axes[row, 0]
    ax.hist(ms_tr, bins=25, color="#5B8DB8", edgecolor="white", lw=0.4)
    ax.axvline(thr, color=CT, ls="--", lw=1.6,
               label=f"10th pct = {thr:.3f}")
    ax.set_xlabel("Mean Tanimoto similarity (k=5 NN)")
    ax.set_ylabel("Count")
    ax.set_title(f"{labels[row][0]} {title}\nTraining set",
                 fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=8)

    # test distribution by AD
    ax = axes[row, 1]
    edges = np.histogram_bin_edges(ms_te, bins=20)
    ax.hist(ms_te[ad],  bins=edges, color=CI, alpha=0.85,
            label=f"Within AD (n={ad.sum()})", edgecolor="white", lw=0.4)
    ax.hist(ms_te[~ad], bins=edges, color=CO, alpha=0.85,
            label=f"Outside AD (n={(~ad).sum()})", edgecolor="white", lw=0.4)
    ax.axvline(thr, color=CT, ls="--", lw=1.6)
    ax.set_xlabel("Mean Tanimoto similarity (k=5 NN)")
    ax.set_ylabel("Count")
    ax.set_title(f"{labels[row][1]} Test set",
                 fontweight="bold", fontsize=9.5)
    ax.legend(fontsize=8)

    # sorted per-compound
    ax = axes[row, 2]
    idx = np.argsort(ms_te)
    ss  = ms_te[idx]
    cc  = np.where(ad[idx], CI, CO)
    ax.scatter(np.arange(len(ss)), ss, c=cc, s=16, alpha=0.7, zorder=2)
    ax.axhline(thr, color=CT, ls="--", lw=1.6)
    ax.set_xlabel("Compounds (sorted)")
    ax.set_ylabel("Mean Tanimoto similarity (k=5 NN)")
    ax.set_title(f"{labels[row][2]} Per-compound similarity",
                 fontweight="bold", fontsize=9.5)
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=CI, ms=6,
               label="Within AD"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=CO, ms=6,
               label="Outside AD"),
        Line2D([0],[0], color=CT, ls="--", lw=1.6,
               label=f"Threshold = {thr:.3f}"),
    ], fontsize=7.5)

fig.suptitle(
    "Applicability domain — sweetness intensity regression\n"
    "ECFP4 Tanimoto k-NN (k=5), threshold = 10th percentile",
    fontsize=11, fontweight="bold", y=0.99)

plt.savefig("ad_regression.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved ad_regression.png")
