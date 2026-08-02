# %%
# %%
"""
D-Tagatose: DFT/B3LYP/6-311++G(d,p) IR Frequency Scaling Factor Analysis
==========================================================================

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

# ─── FILE PATHS ────────────────────────────────────────────────────────────────
THEO_FILE = "d-tagatose-ir.txt"
EXP_FILE  = "TAG.CSV"
OUTPUT    = "d-tagatose_IR.png"
# ──────────────────────────────────────────────────────────────────────────────


# STEP 1 — LOAD & SORT
theo_raw = np.loadtxt(THEO_FILE)
exp_raw  = np.loadtxt(EXP_FILE, delimiter=",")
theo_raw = theo_raw[np.argsort(theo_raw[:, 0])]
exp_raw  = exp_raw [np.argsort(exp_raw [:, 0])]
exp_wn   = exp_raw[:, 0]
exp_y    = exp_raw[:, 1]


# STEP 2 — Y-AXIS DIAGNOSIS & %T -> ABSORBANCE
y_min, y_max = exp_y.min(), exp_y.max()
silent_mean  = exp_y[(exp_wn > 1800) & (exp_wn < 2800)].mean()
co_min       = exp_y[(exp_wn >  800) & (exp_wn < 1200)].min()

is_transmittance = (0 <= y_min and y_max <= 100
                    and silent_mean > 85 and co_min < 60)

print("Y-AXIS DIAGNOSIS")
print(f"  Range         : {y_min:.2f} - {y_max:.2f}")
print(f"  Silent region : {silent_mean:.2f}  (threshold > 85 for %T)")
print(f"  C-O band min  : {co_min:.2f}  (threshold < 60 for %T)")
print(f"  Verdict       : {'%Transmittance confirmed' if is_transmittance else 'WARNING - check input'}")
print()
if not is_transmittance:
    raise ValueError("Y-axis does not appear to be %Transmittance.")

exp_T        = np.clip(exp_y, 0.01, 100)
exp_abs      = -np.log10(exp_T / 100)
exp_abs_norm = exp_abs / exp_abs.max()


# STEP 3 — SAVITZKY-GOLAY 
SG_WINDOW = 21
SG_POLY   = 3
sg_smooth = savgol_filter(exp_abs, window_length=SG_WINDOW, polyorder=SG_POLY)
noise_std = np.std(exp_abs[(exp_wn > 1800) & (exp_wn < 2800)])


# STEP 4 — EXPERIMENTAL PEAK DETECTION
PROMINENCE = 0.010
DISTANCE   = 30
exp_pk_idx, exp_pk_props = find_peaks(sg_smooth, prominence=PROMINENCE, distance=DISTANCE)
exp_pk_wn  = exp_wn [exp_pk_idx]
exp_pk_abs = exp_abs[exp_pk_idx]
n_exp_detected = len(exp_pk_idx)

print(f"EXPERIMENTAL PEAKS DETECTED: n = {n_exp_detected}")
print(f"  SG: window={SG_WINDOW}, polyorder={SG_POLY}")
print(f"  find_peaks: prominence >= {PROMINENCE}, distance >= {DISTANCE} pts")
print()


# STEP 5 — THEORETICAL PEAKS (Gaussian, 400-4000 cm-1)
mask_theo    = (theo_raw[:, 0] >= 400) & (theo_raw[:, 0] <= 4000)
theo_wn_all  = theo_raw[mask_theo, 0]
theo_int_all = theo_raw[mask_theo, 1]
theo_int_norm = theo_int_all / theo_int_all.max()
n_theo_total  = len(theo_wn_all)


# STEP 6 — AUTOMATIC PEAK MATCHING
TOLERANCE = 30
priority_order = np.argsort(-theo_int_all)
used_exp_idx   = set()
matched_pairs  = []

for idx in priority_order:
    tw = theo_wn_all[idx]
    ti = theo_int_all[idx]
    diffs = np.abs(exp_pk_wn - tw)
    j = int(np.argmin(diffs))
    if diffs[j] <= TOLERANCE and j not in used_exp_idx:
        matched_pairs.append({
            "theo_wn"  : tw,
            "exp_wn"   : exp_pk_wn[j],
            "theo_int" : ti,
            "exp_abs"  : exp_pk_abs[j],
            "delta"    : exp_pk_wn[j] - tw,
            "sf"       : exp_pk_wn[j] / tw,
        })
        used_exp_idx.add(j)

matched_pairs.sort(key=lambda x: x["theo_wn"])
theo_v         = np.array([p["theo_wn"]  for p in matched_pairs])
exp_v          = np.array([p["exp_wn"]   for p in matched_pairs])
sfs            = np.array([p["sf"]       for p in matched_pairs])
theo_int_match = np.array([p["theo_int"] for p in matched_pairs])
n_matched      = len(matched_pairs)
pct_theo_matched = 100 * n_matched / n_theo_total
pct_exp_matched  = 100 * n_matched / n_exp_detected


# STEP 7 — SCALING FACTOR STATISTICS
SF_mean   = sfs.mean()
SF_std    = sfs.std(ddof=1)
SF_median = np.median(sfs)
n_pairs   = len(sfs)
SF_95lo   = SF_mean - 1.96 * SF_std / np.sqrt(n_pairs)
SF_95hi   = SF_mean + 1.96 * SF_std / np.sqrt(n_pairs)

SF        = np.sum(theo_v * exp_v) / np.sum(theo_v**2)
residuals = exp_v - SF * theo_v
SS_res    = np.sum(residuals**2)
SS_tot    = np.sum((exp_v - exp_v.mean())**2)
R2        = 1 - SS_res / SS_tot
MAE       = np.mean(np.abs(residuals))
RMSE      = np.sqrt(np.mean(residuals**2))
slope_f, intercept_f, r_f, p_f, _ = linregress(theo_v, exp_v)

print("=" * 65)
print("TABLE S1 — Automatic peak matching results")
print(f"  Wavenumber range            : 400-4000 cm-1")
print(f"  Theoretical peaks in range  : {n_theo_total}")
print(f"  Experimental peaks detected : {n_exp_detected}")
print(f"  Matched pairs               : {n_matched} "
      f"({pct_theo_matched:.0f}% of theoretical, {pct_exp_matched:.0f}% of experimental)")
print(f"  Tolerance                   : +/-{TOLERANCE} cm-1")
print("=" * 65)
print(f"{'nu_calc (cm-1)':>16}  {'nu_exp (cm-1)':>14}  {'Delta_nu':>10}  {'SF_i':>10}")
print("-" * 58)
for p in matched_pairs:
    print(f"{p['theo_wn']:>16.1f}  {p['exp_wn']:>14.1f}  "
          f"{p['delta']:>+10.1f}  {p['sf']:>10.5f}")
print("-" * 58)
print(f"\nSCALING FACTOR SUMMARY")
print(f"  Method A mean ratio   : {SF_mean:.5f}   SD = {SF_std:.5f}")
print(f"  95% CI (mean)         : [{SF_95lo:.4f}, {SF_95hi:.4f}]")
print(f"  Median SF             : {SF_median:.5f}")
print(f"  Method B regression   : {SF:.5f}")
print(f"  R2                    : {R2:.6f}")
print(f"  MAE                   : {MAE:.2f} cm-1")
print(f"  RMSE                  : {RMSE:.2f} cm-1")
print()


# STEP 8 — LORENTZIAN BROADENING
def lorentzian(wn_grid, centers, intensities, fwhm=12.0):
    spec = np.zeros_like(wn_grid, dtype=float)
    for c, i in zip(centers, intensities):
        spec += i / (1.0 + ((wn_grid - c) / (fwhm / 2.0))**2)
    return spec / spec.max() if spec.max() > 0 else spec

wn_grid    = np.linspace(400, 4000, 5000)
theo_broad = lorentzian(wn_grid, theo_wn_all, theo_int_norm, fwhm=12)


# STEP 9 — FIVE-PANEL FIGURE 
matplotlib.rcParams.update({
    "font.family"     : "sans-serif",
    "font.size"       : 10,
    "axes.labelsize"  : 11,
    "axes.titlesize"  : 10.5,
    "xtick.labelsize" : 9.5,
    "ytick.labelsize" : 9.5,
    "legend.fontsize" : 9,
    "axes.labelweight": "bold",
    "figure.dpi"      : 300,
})

C_EXP   = "#1a6faf"
C_THEO  = "#c0392b"
C_MATCH = "#d4820a"
C_GRID  = "#efefef"
C_ZERO  = "#aaaaaa"

def style_ax(ax, xlabel="Wavenumber (cm\u207b\u00b9)", ylabel=None):
    ax.set_facecolor("white")
    ax.tick_params(which="major", direction="out", length=4, width=0.9,
                   color="#444444", labelcolor="#111111")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=2, width=0.5,
                   color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888");   ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#888888"); ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(True, which="major", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)

fig = plt.figure(figsize=(14, 18), facecolor="white")
gs_outer = gridspec.GridSpec(
    4, 1, figure=fig, hspace=0.54,
    top=0.945, bottom=0.048, left=0.09, right=0.965,
    height_ratios=[1, 1, 1, 1.05]
)
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[3], wspace=0.38, width_ratios=[1.5, 1]
)
ax1 = fig.add_subplot(gs_outer[0])
ax2 = fig.add_subplot(gs_outer[1])
ax3 = fig.add_subplot(gs_outer[2])
ax4 = fig.add_subplot(gs_bottom[0])
ax5 = fig.add_subplot(gs_bottom[1])


# Panel a — Experimental FTIR
style_ax(ax1, ylabel="Normalized absorbance")
ax1.plot(exp_wn, exp_abs_norm, color=C_EXP, lw=1.0,
         label="Experimental FTIR spectrum (%Transmittance converted to absorbance)")
for w, a in zip(exp_pk_wn, exp_pk_abs / exp_abs.max()):
    if a > 0.07:
        ax1.text(w, a + 0.04, f"{w:.0f}", fontsize=6, color=C_EXP,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax1.set_xlim(400, 4000)
ax1.set_ylim(-0.04, 1.50)
ax1.set_title("(a)  Experimental FTIR Spectrum — D-Tagatose",
              fontweight="bold", loc="left", pad=5)
ax1.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb", bbox_to_anchor=(1.0, 0.88))
ax1.text(0.005, 0.97,
         f"Conversion: A = \u2212log\u2081\u2080(T/100)  |  "
         f"SG smoothing (w = {SG_WINDOW}, p = {SG_POLY}) for peak detection only  |  "
         f"{n_exp_detected} peaks detected (prominence \u2265 {PROMINENCE} A, "
         f"distance \u2265 {DISTANCE} pts)  |  Raw data plotted.",
         transform=ax1.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel b — DFT stick spectrum
style_ax(ax2, ylabel="Normalized IR intensity")
ax2.vlines(theo_wn_all, 0, theo_int_norm, color=C_THEO, lw=1.5, alpha=0.85,
           label="DFT/B3LYP/6-311++G(d,p) calculated IR spectrum (unscaled)")
for w, i in zip(theo_wn_all, theo_int_norm):
    if i > 0.12:
        ax2.text(w, i + 0.03, f"{w:.0f}", fontsize=6, color=C_THEO,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax2.set_xlim(400, 4000)
ax2.set_ylim(-0.04, 1.50)
ax2.set_title("(b)  Calculated IR Spectrum — DFT/B3LYP/6-311++G(d,p)",
              fontweight="bold", loc="left", pad=5)
ax2.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax2.text(0.005, 0.97,
         f"Gaussian frequency output  |  {n_theo_total} modes in 400\u20134000 cm\u207b\u00b9  |  Unscaled",
         transform=ax2.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel c — Overlay + matched peaks
style_ax(ax3, ylabel="Normalized absorbance / intensity")
ax3.plot(exp_wn, exp_abs_norm, color=C_EXP, lw=1.0, alpha=0.92,
         label="Experimental (normalized absorbance)")
ax3.plot(wn_grid, theo_broad * 0.80, color=C_THEO, lw=0.9, alpha=0.65,
         label="Calculated (Lorentzian-broadened, FWHM = 12 cm\u207b\u00b9, for visual comparison only)")
for p in matched_pairs:
    idx_e = int(np.argmin(np.abs(exp_wn - p["exp_wn"])))
    y_e   = exp_abs_norm[idx_e]
    ew, tw = p["exp_wn"], p["theo_wn"]
    ax3.plot([ew, ew],  [y_e, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot([ew, tw],  [-0.04, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot(ew,  y_e,  "o", color=C_EXP,  ms=4,   zorder=6, mew=0.6, mec="white")
    ax3.plot(tw, -0.04, "s", color=C_THEO, ms=3.5, zorder=6, mew=0.6, mec="white")
ax3.axhline(-0.04, color="#cccccc", lw=0.6, ls=":")
ax3.text(3950, -0.04, "calc.", color=C_THEO, fontsize=7,
         va="center", ha="right", fontweight="bold")
ax3.set_xlim(400, 4000)
ax3.set_ylim(-0.10, 1.30)
ax3.set_title(
    f"(c)  Spectral Overlay and Peak Correspondence  |  "
    f"n = {n_matched} matched pairs  |  "
    f"400\u20134000 cm\u207b\u00b9  |  "
    f"{pct_theo_matched:.0f}% of calculated modes matched",
    fontweight="bold", loc="left", pad=5)
ax3.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax3.text(0.005, 0.97,
         f"Matching: nearest-neighbor within \u00b1{TOLERANCE} cm\u207b\u00b9; "
         "priority by descending calculated intensity; each experimental peak used once.\n"
         "Lorentzian broadening for visual comparison only — "
         "scaling factor derived from discrete peak positions.",
         transform=ax3.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel d — Correlation
style_ax(ax4,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="\u03bd$_{exp}$ (cm$^{-1}$)")
sizes = 30 + 140 * (theo_int_match / theo_int_match.max())
sc = ax4.scatter(theo_v, exp_v, s=sizes, c=theo_int_match,
                 cmap="YlOrRd", zorder=5, edgecolors="#222222",
                 lw=0.5, alpha=0.88,
                 label=f"Matched peaks (n = {n_matched}, size/color \u221d IR intensity)")
lim_min, lim_max = 380, 3050
x_fit = np.linspace(lim_min, lim_max, 500)
ax4.plot(x_fit, x_fit, "--", color="#888888", lw=1.2, label="1:1 reference")
ax4.plot(x_fit, SF * x_fit, "-", color=C_THEO, lw=2.0,
         label=f"Origin-forced: slope = {SF:.4f}")
cb = plt.colorbar(sc, ax=ax4, pad=0.01, shrink=0.8)
cb.set_label("Calculated\nintensity (km/mol)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
stats = (f"SF = {SF:.5f}  (origin regression)\n"
         f"R\u00b2 = {R2:.6f}\n"
         f"MAE = {MAE:.2f} cm\u207b\u00b9   RMSE = {RMSE:.2f} cm\u207b\u00b9\n"
         f"Mean SF = {SF_mean:.5f}   SD = {SF_std:.5f}\n"
         f"95% CI [{SF_95lo:.4f}, {SF_95hi:.4f}]")
ax4.text(0.03, 0.97, stats, transform=ax4.transAxes,
         fontsize=8.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f0f4ff", ec="#9999cc", lw=0.8))
ax4.set_xlim(lim_min, lim_max)
ax4.set_ylim(lim_min, lim_max)
ax4.set_title("(d)  Calculated vs. Experimental Wavenumber Correlation",
              fontweight="bold", loc="left", pad=5)
ax4.legend(loc="lower right", frameon=True, edgecolor="#bbbbbb")


# Panel e — Residuals
style_ax(ax5,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="Residual (cm$^{-1}$)")
ax5.scatter(theo_v, residuals, s=sizes, c=theo_int_match,
            cmap="YlOrRd", zorder=5, edgecolors="#222222", lw=0.5, alpha=0.88)
ax5.axhline(0,     color=C_ZERO,    lw=1.1, ls="--", zorder=3)
ax5.axhline( MAE,  color="#2ecc71", lw=1.0, ls=":",  label=f"\u00b1MAE = {MAE:.1f} cm\u207b\u00b9")
ax5.axhline(-MAE,  color="#2ecc71", lw=1.0, ls=":")
ax5.axhline( RMSE, color="#e74c3c", lw=1.0, ls="-.", label=f"\u00b1RMSE = {RMSE:.1f} cm\u207b\u00b9")
ax5.axhline(-RMSE, color="#e74c3c", lw=1.0, ls="-.")
ax5.set_title("(e)  Residuals: \u03bd$_{exp}$ \u2212 SF\u00b7\u03bd$_{calc}$",
              fontweight="bold", loc="left", pad=5)
ax5.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
rlim = max(np.abs(residuals).max() * 1.3, RMSE * 2)
ax5.set_ylim(-rlim, rlim)
ax5.set_xlim(lim_min, lim_max)


# Supertitle & footnote
fig.suptitle(
    "D-Tagatose: Validation of DFT/B3LYP/6-311++G(d,p) Calculated IR Spectrum\n"
    "against Experimental FTIR Data \u2014 Vibrational Frequency Scaling Factor Analysis",
    fontsize=13, fontweight="bold", y=0.998)

fig.text(0.5, 0.008,
         f"Peak matching: automatic, \u00b1{TOLERANCE} cm\u207b\u00b9 tolerance, "
         f"{n_matched} pairs ({pct_theo_matched:.0f}% of {n_theo_total} "
         "calculated modes, 400\u20134000 cm\u207b\u00b9).  "
         "SG smoothing applied to peak detection only; raw experimental data plotted.  "
         "Lorentzian broadening (FWHM = 12 cm\u207b\u00b9) for visual comparison only.",
         ha="center", fontsize=7.5, color="#444444", style="italic")

plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved -> {OUTPUT}")
# %%
"""
Advantame: DFT/B3LYP/6-311++G(d,p) IR Frequency Scaling Factor Analysis
=========================================================================

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

# ─── FILE PATHS ────────────────────────────────────────────────────────────────
THEO_FILE = "advantame.txt"
EXP_FILE  = "ADV.CSV"
OUTPUT    = "advantame_IR.png"
MOLECULE  = "Advantame"
# ──────────────────────────────────────────────────────────────────────────────


# STEP 1 — LOAD & SORT
theo_raw = np.loadtxt(THEO_FILE)
exp_raw  = np.loadtxt(EXP_FILE, delimiter=",")
theo_raw = theo_raw[np.argsort(theo_raw[:, 0])]
exp_raw  = exp_raw [np.argsort(exp_raw [:, 0])]
exp_wn   = exp_raw[:, 0]
exp_y    = exp_raw[:, 1]


# STEP 2 — Y-AXIS DIAGNOSIS & %T -> ABSORBANCE
y_min, y_max = exp_y.min(), exp_y.max()
silent_mean  = exp_y[(exp_wn > 1800) & (exp_wn < 2400)].mean()
co_min       = exp_y[(exp_wn >  800) & (exp_wn < 1200)].min()

is_transmittance = (0 <= y_min and y_max <= 100
                    and silent_mean > 85 and co_min < 60)

print("Y-AXIS DIAGNOSIS")
print(f"  Range         : {y_min:.2f} - {y_max:.2f}")
print(f"  Silent region : {silent_mean:.2f}  (threshold > 85 for %T)")
print(f"  C-O band min  : {co_min:.2f}  (threshold < 60 for %T)")
print(f"  Verdict       : {'%Transmittance confirmed' if is_transmittance else 'WARNING - check input'}")
print()
if not is_transmittance:
    raise ValueError("Y-axis does not appear to be %Transmittance.")

exp_T        = np.clip(exp_y, 0.01, 100)
exp_abs      = -np.log10(exp_T / 100)
exp_abs_norm = exp_abs / exp_abs.max()


# STEP 3 — SAVITZKY-GOLAY 
SG_WINDOW = 21
SG_POLY   = 3
sg_smooth = savgol_filter(exp_abs, window_length=SG_WINDOW, polyorder=SG_POLY)
noise_std = np.std(exp_abs[(exp_wn > 1800) & (exp_wn < 2400)])


# STEP 4 — EXPERIMENTAL PEAK DETECTION
PROMINENCE = 0.010
DISTANCE   = 30
exp_pk_idx, _ = find_peaks(sg_smooth, prominence=PROMINENCE, distance=DISTANCE)
exp_pk_wn  = exp_wn [exp_pk_idx]
exp_pk_abs = exp_abs[exp_pk_idx]
n_exp_detected = len(exp_pk_idx)

print(f"EXPERIMENTAL PEAKS DETECTED: n = {n_exp_detected}")
print(f"  SG: window={SG_WINDOW}, polyorder={SG_POLY}")
print(f"  find_peaks: prominence >= {PROMINENCE}, distance >= {DISTANCE} pts")
print()


# STEP 5 — THEORETICAL PEAKS (Gaussian, 400-4000 cm-1)
mask_theo    = (theo_raw[:, 0] >= 400) & (theo_raw[:, 0] <= 4000)
theo_wn_all  = theo_raw[mask_theo, 0]
theo_int_all = theo_raw[mask_theo, 1]
theo_int_norm = theo_int_all / theo_int_all.max()
n_theo_total  = len(theo_wn_all)


# STEP 6 — AUTOMATIC PEAK MATCHING
TOLERANCE = 30
priority_order = np.argsort(-theo_int_all)
used_exp_idx   = set()
matched_pairs  = []

for idx in priority_order:
    tw = theo_wn_all[idx]
    ti = theo_int_all[idx]
    diffs = np.abs(exp_pk_wn - tw)
    j = int(np.argmin(diffs))
    if diffs[j] <= TOLERANCE and j not in used_exp_idx:
        matched_pairs.append({
            "theo_wn"  : tw,
            "exp_wn"   : exp_pk_wn[j],
            "theo_int" : ti,
            "exp_abs"  : exp_pk_abs[j],
            "delta"    : exp_pk_wn[j] - tw,
            "sf"       : exp_pk_wn[j] / tw,
        })
        used_exp_idx.add(j)

matched_pairs.sort(key=lambda x: x["theo_wn"])
theo_v         = np.array([p["theo_wn"]  for p in matched_pairs])
exp_v          = np.array([p["exp_wn"]   for p in matched_pairs])
sfs            = np.array([p["sf"]       for p in matched_pairs])
theo_int_match = np.array([p["theo_int"] for p in matched_pairs])
n_matched      = len(matched_pairs)
pct_theo_matched = 100 * n_matched / n_theo_total
pct_exp_matched  = 100 * n_matched / n_exp_detected


# STEP 7 — SCALING FACTOR STATISTICS
SF_mean   = sfs.mean()
SF_std    = sfs.std(ddof=1)
SF_median = np.median(sfs)
n_pairs   = len(sfs)
SF_95lo   = SF_mean - 1.96 * SF_std / np.sqrt(n_pairs)
SF_95hi   = SF_mean + 1.96 * SF_std / np.sqrt(n_pairs)

SF        = np.sum(theo_v * exp_v) / np.sum(theo_v**2)
residuals = exp_v - SF * theo_v
SS_res    = np.sum(residuals**2)
SS_tot    = np.sum((exp_v - exp_v.mean())**2)
R2        = 1 - SS_res / SS_tot
MAE       = np.mean(np.abs(residuals))
RMSE      = np.sqrt(np.mean(residuals**2))
slope_f, intercept_f, r_f, p_f, _ = linregress(theo_v, exp_v)

print("=" * 65)
print(f"TABLE S — Automatic peak matching results ({MOLECULE})")
print(f"  Wavenumber range            : 400-4000 cm-1")
print(f"  Theoretical peaks in range  : {n_theo_total}")
print(f"  Experimental peaks detected : {n_exp_detected}")
print(f"  Matched pairs               : {n_matched} "
      f"({pct_theo_matched:.0f}% of theoretical, {pct_exp_matched:.0f}% of experimental)")
print(f"  Tolerance                   : +/-{TOLERANCE} cm-1")
print("=" * 65)
print(f"{'nu_calc (cm-1)':>16}  {'nu_exp (cm-1)':>14}  {'Delta_nu':>10}  {'SF_i':>10}")
print("-" * 58)
for p in matched_pairs:
    print(f"{p['theo_wn']:>16.1f}  {p['exp_wn']:>14.1f}  "
          f"{p['delta']:>+10.1f}  {p['sf']:>10.5f}")
print("-" * 58)
print(f"\nSCALING FACTOR SUMMARY")
print(f"  Method A mean ratio   : {SF_mean:.5f}   SD = {SF_std:.5f}")
print(f"  95% CI (mean)         : [{SF_95lo:.4f}, {SF_95hi:.4f}]")
print(f"  Median SF             : {SF_median:.5f}")
print(f"  Method B regression   : {SF:.5f}")
print(f"  R2                    : {R2:.6f}")
print(f"  MAE                   : {MAE:.2f} cm-1")
print(f"  RMSE                  : {RMSE:.2f} cm-1")
print()


# STEP 8 — LORENTZIAN BROADENING
def lorentzian(wn_grid, centers, intensities, fwhm=12.0):
    spec = np.zeros_like(wn_grid, dtype=float)
    for c, i in zip(centers, intensities):
        spec += i / (1.0 + ((wn_grid - c) / (fwhm / 2.0))**2)
    return spec / spec.max() if spec.max() > 0 else spec

wn_grid    = np.linspace(400, 4000, 5000)
theo_broad = lorentzian(wn_grid, theo_wn_all, theo_int_norm, fwhm=12)


# STEP 9 — FIVE-PANEL PUBLICATION FIGURE (300 DPI)
matplotlib.rcParams.update({
    "font.family"     : "sans-serif",
    "font.size"       : 10,
    "axes.labelsize"  : 11,
    "axes.titlesize"  : 10.5,
    "xtick.labelsize" : 9.5,
    "ytick.labelsize" : 9.5,
    "legend.fontsize" : 9,
    "axes.labelweight": "bold",
    "figure.dpi"      : 300,
})

C_EXP   = "#1a6faf"
C_THEO  = "#c0392b"
C_MATCH = "#d4820a"
C_GRID  = "#efefef"
C_ZERO  = "#aaaaaa"

def style_ax(ax, xlabel="Wavenumber (cm\u207b\u00b9)", ylabel=None):
    ax.set_facecolor("white")
    ax.tick_params(which="major", direction="out", length=4, width=0.9,
                   color="#444444", labelcolor="#111111")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=2, width=0.5,
                   color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888");   ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#888888"); ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(True, which="major", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)

fig = plt.figure(figsize=(14, 18), facecolor="white")
gs_outer = gridspec.GridSpec(
    4, 1, figure=fig, hspace=0.54,
    top=0.945, bottom=0.048, left=0.09, right=0.965,
    height_ratios=[1, 1, 1, 1.05]
)
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[3], wspace=0.38, width_ratios=[1.5, 1]
)
ax1 = fig.add_subplot(gs_outer[0])
ax2 = fig.add_subplot(gs_outer[1])
ax3 = fig.add_subplot(gs_outer[2])
ax4 = fig.add_subplot(gs_bottom[0])
ax5 = fig.add_subplot(gs_bottom[1])


# Panel a — Experimental FTIR
style_ax(ax1, ylabel="Normalized absorbance")
ax1.plot(exp_wn, exp_abs_norm, color=C_EXP, lw=1.0,
         label="Experimental FTIR spectrum (%Transmittance converted to absorbance)")
for w, a in zip(exp_pk_wn, exp_pk_abs / exp_abs.max()):
    if a > 0.07:
        ax1.text(w, a + 0.04, f"{w:.0f}", fontsize=6, color=C_EXP,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax1.set_xlim(400, 4000)
ax1.set_ylim(-0.04, 1.50)
ax1.set_title(f"(a)  Experimental FTIR Spectrum \u2014 {MOLECULE}",
              fontweight="bold", loc="left", pad=5)
ax1.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb", bbox_to_anchor=(1.0, 0.88))
ax1.text(0.005, 0.97,
         f"Conversion: A = \u2212log\u2081\u2080(T/100)  |  "
         f"SG smoothing (w = {SG_WINDOW}, p = {SG_POLY}) for peak detection only  |  "
         f"{n_exp_detected} peaks detected (prominence \u2265 {PROMINENCE} A, "
         f"distance \u2265 {DISTANCE} pts)  |  Raw data plotted.",
         transform=ax1.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel b — DFT stick spectrum
style_ax(ax2, ylabel="Normalized IR intensity")
ax2.vlines(theo_wn_all, 0, theo_int_norm, color=C_THEO, lw=1.5, alpha=0.85,
           label="DFT/B3LYP/6-311++G(d,p) calculated IR spectrum (unscaled)")
for w, i in zip(theo_wn_all, theo_int_norm):
    if i > 0.12:
        ax2.text(w, i + 0.03, f"{w:.0f}", fontsize=6, color=C_THEO,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax2.set_xlim(400, 4000)
ax2.set_ylim(-0.04, 1.50)
ax2.set_title(f"(b)  Calculated IR Spectrum \u2014 DFT/B3LYP/6-311++G(d,p)",
              fontweight="bold", loc="left", pad=5)
ax2.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax2.text(0.005, 0.97,
         f"Gaussian frequency output  |  {n_theo_total} modes in 400\u20134000 cm\u207b\u00b9  |  Unscaled",
         transform=ax2.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel c — Overlay + matched peaks
style_ax(ax3, ylabel="Normalized absorbance / intensity")
ax3.plot(exp_wn, exp_abs_norm, color=C_EXP, lw=1.0, alpha=0.92,
         label="Experimental (normalized absorbance)")
ax3.plot(wn_grid, theo_broad * 0.80, color=C_THEO, lw=0.9, alpha=0.65,
         label="Calculated (Lorentzian-broadened, FWHM = 12 cm\u207b\u00b9, for visual comparison only)")
for p in matched_pairs:
    idx_e = int(np.argmin(np.abs(exp_wn - p["exp_wn"])))
    y_e   = exp_abs_norm[idx_e]
    ew, tw = p["exp_wn"], p["theo_wn"]
    ax3.plot([ew, ew],  [y_e, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot([ew, tw],  [-0.04, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot(ew,  y_e,  "o", color=C_EXP,  ms=4,   zorder=6, mew=0.6, mec="white")
    ax3.plot(tw, -0.04, "s", color=C_THEO, ms=3.5, zorder=6, mew=0.6, mec="white")
ax3.axhline(-0.04, color="#cccccc", lw=0.6, ls=":")
ax3.text(3950, -0.04, "calc.", color=C_THEO, fontsize=7,
         va="center", ha="right", fontweight="bold")
ax3.set_xlim(400, 4000)
ax3.set_ylim(-0.10, 1.30)
ax3.set_title(
    f"(c)  Spectral Overlay and Peak Correspondence  |  "
    f"n = {n_matched} matched pairs  |  "
    f"400\u20134000 cm\u207b\u00b9  |  "
    f"{pct_theo_matched:.0f}% of calculated modes matched",
    fontweight="bold", loc="left", pad=5)
ax3.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax3.text(0.005, 0.97,
         f"Matching: nearest-neighbor within \u00b1{TOLERANCE} cm\u207b\u00b9; "
         "priority by descending calculated intensity; each experimental peak used once.\n"
         "Lorentzian broadening for visual comparison only \u2014 "
         "scaling factor derived from discrete peak positions.",
         transform=ax3.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel d — Correlation
style_ax(ax4,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="\u03bd$_{exp}$ (cm$^{-1}$)")
sizes = 30 + 140 * (theo_int_match / theo_int_match.max())
sc = ax4.scatter(theo_v, exp_v, s=sizes, c=theo_int_match,
                 cmap="YlOrRd", zorder=5, edgecolors="#222222",
                 lw=0.5, alpha=0.88,
                 label=f"Matched peaks (n = {n_matched}, size/color \u221d IR intensity)")
lim_min, lim_max = 380, 3700
x_fit = np.linspace(lim_min, lim_max, 500)
ax4.plot(x_fit, x_fit, "--", color="#888888", lw=1.2, label="1:1 reference")
ax4.plot(x_fit, SF * x_fit, "-", color=C_THEO, lw=2.0,
         label=f"Origin-forced: slope = {SF:.4f}")
cb = plt.colorbar(sc, ax=ax4, pad=0.01, shrink=0.8)
cb.set_label("Calculated\nintensity (km/mol)", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
stats = (f"SF = {SF:.5f}  (origin regression)\n"
         f"R\u00b2 = {R2:.6f}\n"
         f"MAE = {MAE:.2f} cm\u207b\u00b9   RMSE = {RMSE:.2f} cm\u207b\u00b9\n"
         f"Mean SF = {SF_mean:.5f}   SD = {SF_std:.5f}\n"
         f"95% CI [{SF_95lo:.4f}, {SF_95hi:.4f}]")
ax4.text(0.03, 0.97, stats, transform=ax4.transAxes,
         fontsize=8.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f0f4ff", ec="#9999cc", lw=0.8))
ax4.set_xlim(lim_min, lim_max)
ax4.set_ylim(lim_min, lim_max)
ax4.set_title("(d)  Calculated vs. Experimental Wavenumber Correlation",
              fontweight="bold", loc="left", pad=5)
ax4.legend(loc="lower right", frameon=True, edgecolor="#bbbbbb")


# Panel e — Residuals
style_ax(ax5,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="Residual (cm$^{-1}$)")
ax5.scatter(theo_v, residuals, s=sizes, c=theo_int_match,
            cmap="YlOrRd", zorder=5, edgecolors="#222222", lw=0.5, alpha=0.88)
ax5.axhline(0,     color=C_ZERO,    lw=1.1, ls="--", zorder=3)
ax5.axhline( MAE,  color="#2ecc71", lw=1.0, ls=":",  label=f"\u00b1MAE = {MAE:.1f} cm\u207b\u00b9")
ax5.axhline(-MAE,  color="#2ecc71", lw=1.0, ls=":")
ax5.axhline( RMSE, color="#e74c3c", lw=1.0, ls="-.", label=f"\u00b1RMSE = {RMSE:.1f} cm\u207b\u00b9")
ax5.axhline(-RMSE, color="#e74c3c", lw=1.0, ls="-.")
ax5.set_title("(e)  Residuals: \u03bd$_{exp}$ \u2212 SF\u00b7\u03bd$_{calc}$",
              fontweight="bold", loc="left", pad=5)
ax5.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
rlim = max(np.abs(residuals).max() * 1.3, RMSE * 2)
ax5.set_ylim(-rlim, rlim)
ax5.set_xlim(lim_min, lim_max)


# Supertitle & footnote
fig.suptitle(
    f"{MOLECULE}: Validation of DFT/B3LYP/6-311++G(d,p) Calculated IR Frequencies\n"
    "against Experimental FTIR Data \u2014 Vibrational Frequency Scaling Factor Analysis",
    fontsize=13, fontweight="bold", y=0.998)

fig.text(0.5, 0.008,
         f"Peak matching: automatic, \u00b1{TOLERANCE} cm\u207b\u00b9 tolerance, "
         f"{n_matched} pairs ({pct_theo_matched:.0f}% of {n_theo_total} "
         "calculated modes, 400\u20134000 cm\u207b\u00b9).  "
         "SG smoothing applied to peak detection only; raw experimental data plotted.  "
         "Lorentzian broadening (FWHM = 12 cm\u207b\u00b9) for visual comparison only.",
         ha="center", fontsize=7.5, color="#444444", style="italic")

plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved -> {OUTPUT}")
# %%
"""
Advantame: DFT/B3LYP/6-311++G(d,p) Raman Vibrational Frequency Scaling Factor Analysis
==========================================================================================

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

# ─── FILE PATHS ────────────────────────────────────────────────────────────────
THEO_FILE = "advantame_raman.txt"
EXP_FILE  = "ADV 50 x 532nm 2400g 50p.txt"
OUTPUT    = "advantame_raman.png"
MOLECULE  = "Advantame"
# ──────────────────────────────────────────────────────────────────────────────


# STEP 1 — LOAD & SORT
theo_raw = np.loadtxt(THEO_FILE)
exp_raw  = np.loadtxt(EXP_FILE)

theo_raw = theo_raw[np.argsort(theo_raw[:, 0])]
exp_raw  = exp_raw [np.argsort(exp_raw [:, 0])]

exp_wn   = exp_raw[:, 0]
exp_y    = exp_raw[:, 1]
exp_norm = exp_y / exp_y.max()


# STEP 2 — THEORETICAL PEAKS (400-3300 cm-1)
mask_theo    = (theo_raw[:, 0] >= 400) & (theo_raw[:, 0] <= 3300)
theo_wn_all  = theo_raw[mask_theo, 0]
theo_int_all = theo_raw[mask_theo, 1]   
theo_norm    = theo_int_all / theo_int_all.max()
n_theo_total = len(theo_wn_all)

print(f"Theoretical peaks (400-3300 cm-1): n = {n_theo_total}")


# STEP 3 — SAVITZKY-GOLAY 
SG_WINDOW = 21
SG_POLY   = 3
sg_smooth = savgol_filter(exp_y, window_length=SG_WINDOW, polyorder=SG_POLY)

noise_std = np.std(exp_y[(exp_wn > 1800) & (exp_wn < 2200)])
print(f"Noise estimate (1800-2200 cm-1 region): {noise_std:.1f} counts")


# STEP 4 — EXPERIMENTAL PEAK DETECTION
PROMINENCE_FRAC = 0.03
PROMINENCE      = PROMINENCE_FRAC * exp_y.max()
DISTANCE        = 25

exp_pk_idx, _ = find_peaks(sg_smooth, prominence=PROMINENCE, distance=DISTANCE)
exp_pk_wn     = exp_wn[exp_pk_idx]
exp_pk_y      = exp_y [exp_pk_idx]
n_exp_detected = len(exp_pk_idx)

print(f"Experimental peaks detected: n = {n_exp_detected}")
print(f"  SG: window={SG_WINDOW}, polyorder={SG_POLY}")
print(f"  find_peaks: prominence >= {PROMINENCE_FRAC*100:.0f}% of max "
      f"({PROMINENCE:.0f} counts), distance >= {DISTANCE} pts")
print()


# STEP 5 — AUTOMATIC PEAK MATCHING
TOLERANCE = 30

priority_order = np.argsort(-theo_int_all)
used_exp_idx   = set()
matched_pairs  = []

for idx in priority_order:
    tw = theo_wn_all[idx]
    ti = theo_int_all[idx]
    diffs = np.abs(exp_pk_wn - tw)
    j = int(np.argmin(diffs))
    if diffs[j] <= TOLERANCE and j not in used_exp_idx:
        matched_pairs.append({
            "theo_wn"  : tw,
            "exp_wn"   : exp_pk_wn[j],
            "theo_int" : ti,
            "exp_y"    : exp_pk_y[j],
            "delta"    : exp_pk_wn[j] - tw,
            "sf"       : exp_pk_wn[j] / tw,
        })
        used_exp_idx.add(j)

matched_pairs.sort(key=lambda x: x["theo_wn"])

theo_v         = np.array([p["theo_wn"]  for p in matched_pairs])
exp_v          = np.array([p["exp_wn"]   for p in matched_pairs])
sfs            = np.array([p["sf"]       for p in matched_pairs])
theo_int_match = np.array([p["theo_int"] for p in matched_pairs])
n_matched      = len(matched_pairs)

pct_theo = 100 * n_matched / n_theo_total
pct_exp  = 100 * n_matched / n_exp_detected


# STEP 6 — SCALING FACTOR STATISTICS
SF_mean   = sfs.mean()
SF_std    = sfs.std(ddof=1)
SF_median = np.median(sfs)
n_pairs   = len(sfs)
SF_95lo   = SF_mean - 1.96 * SF_std / np.sqrt(n_pairs)
SF_95hi   = SF_mean + 1.96 * SF_std / np.sqrt(n_pairs)

SF        = np.sum(theo_v * exp_v) / np.sum(theo_v**2)
residuals = exp_v - SF * theo_v
SS_res    = np.sum(residuals**2)
SS_tot    = np.sum((exp_v - exp_v.mean())**2)
R2        = 1 - SS_res / SS_tot
MAE       = np.mean(np.abs(residuals))
RMSE      = np.sqrt(np.mean(residuals**2))

slope_f, intercept_f, r_f, _, _ = linregress(theo_v, exp_v)

print("=" * 65)
print(f"TABLE S — Automatic peak matching results ({MOLECULE}, Raman)")
print(f"  Wavenumber range            : 400-3300 cm-1")
print(f"  Theoretical peaks in range  : {n_theo_total}")
print(f"  Experimental peaks detected : {n_exp_detected}")
print(f"  Matched pairs               : {n_matched} "
      f"({pct_theo:.0f}% of theoretical, {pct_exp:.0f}% of experimental)")
print(f"  Tolerance                   : +/-{TOLERANCE} cm-1")
print("=" * 65)
print(f"{'nu_calc (cm-1)':>16}  {'nu_exp (cm-1)':>14}  {'Delta_nu':>10}  {'SF_i':>10}")
print("-" * 58)
for p in matched_pairs:
    print(f"{p['theo_wn']:>16.1f}  {p['exp_wn']:>14.1f}  "
          f"{p['delta']:>+10.1f}  {p['sf']:>10.5f}")
print("-" * 58)
print(f"\nSCALING FACTOR SUMMARY")
print(f"  Method A mean ratio   : {SF_mean:.5f}   SD = {SF_std:.5f}")
print(f"  95% CI (mean)         : [{SF_95lo:.4f}, {SF_95hi:.4f}]")
print(f"  Median SF             : {SF_median:.5f}")
print(f"  Method B regression   : {SF:.5f}")
print(f"  R2                    : {R2:.6f}")
print(f"  MAE                   : {MAE:.2f} cm-1")
print(f"  RMSE                  : {RMSE:.2f} cm-1")
print(f"  Free regression slope : {slope_f:.5f}")
print(f"  Free regression int.  : {intercept_f:.2f} cm-1")
print()


# STEP 7 — LORENTZIAN BROADENING
def lorentzian(wn_grid, centers, intensities, fwhm=10.0):
    spec = np.zeros_like(wn_grid, dtype=float)
    for c, a in zip(centers, intensities):
        spec += a / (1.0 + ((wn_grid - c) / (fwhm / 2.0))**2)
    return spec / spec.max() if spec.max() > 0 else spec

wn_grid    = np.linspace(400, 3300, 5000)
theo_broad = lorentzian(wn_grid, theo_wn_all, theo_norm, fwhm=10)


# STEP 8 — FIVE-PANEL FIGURE 
matplotlib.rcParams.update({
    "font.family"    : "sans-serif",
    "font.size"      : 9,
    "axes.labelsize" : 10,
    "axes.titlesize" : 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.dpi"     : 300,
})

C_EXP   = "#1a6faf"
C_THEO  = "#c0392b"
C_MATCH = "#d4820a"
C_GRID  = "#efefef"
C_ZERO  = "#aaaaaa"

def style_ax(ax, xlabel="Raman shift (cm\u207b\u00b9)", ylabel=None):
    ax.set_facecolor("white")
    ax.tick_params(which="major", direction="out", length=4, width=0.8,
                   color="#555555", labelcolor="#222222")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=2, width=0.5,
                   color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa");   ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#aaaaaa"); ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, which="major", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)

fig = plt.figure(figsize=(14, 18), facecolor="white")
gs_outer = gridspec.GridSpec(
    4, 1, figure=fig, hspace=0.52,
    top=0.945, bottom=0.045, left=0.09, right=0.965,
    height_ratios=[1, 1, 1, 1.05]
)
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[3], wspace=0.38, width_ratios=[1.5, 1]
)
ax1 = fig.add_subplot(gs_outer[0])
ax2 = fig.add_subplot(gs_outer[1])
ax3 = fig.add_subplot(gs_outer[2])
ax4 = fig.add_subplot(gs_bottom[0])
ax5 = fig.add_subplot(gs_bottom[1])

XLO, XHI = 400, 3300


# Panel a — Experimental Raman
style_ax(ax1, ylabel="Normalized intensity")
ax1.plot(exp_wn, exp_norm, color=C_EXP, lw=0.9,
         label="Experimental Raman spectrum (532 nm, 2400 g/mm, normalized)")
for w, y in zip(exp_pk_wn, exp_pk_y / exp_y.max()):
    if y > 0.06:
        ax1.text(w, y + 0.04, f"{w:.0f}", fontsize=5.5, color=C_EXP,
                 ha="center", rotation=90, va="bottom", clip_on=True)
ax1.set_xlim(XLO, XHI)
ax1.set_ylim(-0.04, 1.48)
ax1.set_title(f"(a)  Experimental Raman Spectrum \u2014 {MOLECULE}",
              fontweight="bold", loc="left", pad=5)
ax1.legend(loc="upper right", frameon=True, edgecolor="#cccccc")
ax1.text(0.005, 0.97,
         f"SG smoothing (w = {SG_WINDOW}, p = {SG_POLY}) for peak detection only  |  "
         f"{n_exp_detected} peaks detected "
         f"(prominence \u2265 {PROMINENCE_FRAC*100:.0f}% of max, distance \u2265 {DISTANCE} pts)  |  "
         "Raw data plotted.",
         transform=ax1.transAxes, fontsize=6.5, color="#555555", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#dddddd", lw=0.6))


# Panel b — DFT Raman stick spectrum
style_ax(ax2, xlabel="Raman shift (cm\u207b\u00b9)", ylabel="Normalized Raman intensity")
ax2.vlines(theo_wn_all, 0, theo_norm, color=C_THEO, lw=1.2, alpha=0.80,
           label="DFT/B3LYP/6-311++G(d,p) calculated Raman spectrum (unscaled)")
for w, i in zip(theo_wn_all, theo_norm):
    if i > 0.15:
        ax2.text(w, i + 0.03, f"{w:.0f}", fontsize=5, color=C_THEO,
                 ha="center", rotation=90, va="bottom", clip_on=True)
ax2.set_xlim(XLO, XHI)
ax2.set_ylim(-0.04, 1.48)
ax2.set_title(f"(b)  Calculated Raman Spectrum \u2014 DFT/B3LYP/6-311++G(d,p)",
              fontweight="bold", loc="left", pad=5)
ax2.legend(loc="upper right", frameon=True, edgecolor="#cccccc")
ax2.text(0.005, 0.97,
         f"Gaussian frequency output  |  {n_theo_total} modes in 400\u20133300 cm\u207b\u00b9  |  "
         "Unscaled  |  Y-axis: Raman intensity (normalized)",
         transform=ax2.transAxes, fontsize=6.5, color="#555555", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#dddddd", lw=0.6))


# Panel c — Overlay + matched peaks
style_ax(ax3, xlabel="Raman shift (cm\u207b\u00b9)", ylabel="Normalized intensity")
ax3.plot(exp_wn, exp_norm, color=C_EXP, lw=0.9, alpha=0.92,
         label="Experimental (normalized intensity)")
ax3.plot(wn_grid, theo_broad * 0.80, color=C_THEO, lw=0.85, alpha=0.65,
         label="Calculated (Lorentzian-broadened, FWHM = 10 cm\u207b\u00b9, for visual comparison only)")
for p in matched_pairs:
    idx_e = int(np.argmin(np.abs(exp_wn - p["exp_wn"])))
    y_e   = exp_norm[idx_e]
    ew, tw = p["exp_wn"], p["theo_wn"]
    ax3.plot([ew, ew],  [y_e, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot([ew, tw],  [-0.04, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot(ew,  y_e,  "o", color=C_EXP,  ms=4,   zorder=6, mew=0.6, mec="white")
    ax3.plot(tw, -0.04, "s", color=C_THEO, ms=3.5, zorder=6, mew=0.6, mec="white")
ax3.axhline(-0.04, color="#cccccc", lw=0.6, ls=":")
ax3.text(XHI - 30, -0.04, "calc.", color=C_THEO, fontsize=6.5,
         va="center", ha="right")
ax3.set_xlim(XLO, XHI)
ax3.set_ylim(-0.10, 1.30)
ax3.set_title(
    f"(c)  Spectral Overlay and Peak Correspondence \u2014 "
    f"n = {n_matched} matched pairs  |  "
    f"400\u20133300 cm\u207b\u00b9  |  "
    f"{pct_theo:.0f}% of calculated modes matched",
    fontweight="bold", loc="left", pad=5)
ax3.legend(loc="upper right", frameon=True, edgecolor="#cccccc")
ax3.text(0.005, 0.97,
         f"Matching: nearest-neighbor within \u00b1{TOLERANCE} cm\u207b\u00b9; "
         "priority by descending calculated Raman intensity; each experimental peak used once.\n"
         "Lorentzian broadening for visual comparison only \u2014 "
         "scaling factor derived from discrete peak positions.",
         transform=ax3.transAxes, fontsize=6.5, color="#555555", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#dddddd", lw=0.6))


# Panel d — Correlation
style_ax(ax4,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="\u03bd$_{exp}$ (cm$^{-1}$)")
sizes = 28 + 130 * (theo_int_match / theo_int_match.max())
sc = ax4.scatter(theo_v, exp_v, s=sizes, c=theo_int_match,
                 cmap="YlOrRd", zorder=5, edgecolors="#333333",
                 lw=0.4, alpha=0.88,
                 label=f"Matched peaks (n = {n_matched}, size/color \u221d Raman intensity)")
lim_min, lim_max = 380, 3150
x_fit = np.linspace(lim_min, lim_max, 500)
ax4.plot(x_fit, x_fit, "--", color="#999999", lw=1.1, label="1:1 reference")
ax4.plot(x_fit, SF * x_fit, "-", color=C_THEO, lw=1.8,
         label=f"Origin-forced: slope = {SF:.4f}")
cb = plt.colorbar(sc, ax=ax4, pad=0.01, shrink=0.8)
cb.set_label("Raman intensity", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
stats = (f"SF = {SF:.5f}  (origin regression)\n"
         f"R\u00b2 = {R2:.6f}\n"
         f"MAE = {MAE:.2f} cm\u207b\u00b9   RMSE = {RMSE:.2f} cm\u207b\u00b9\n"
         f"Mean SF = {SF_mean:.5f}   SD = {SF_std:.5f}\n"
         f"95% CI [{SF_95lo:.4f}, {SF_95hi:.4f}]")
ax4.text(0.03, 0.97, stats, transform=ax4.transAxes,
         fontsize=7.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f0f4ff", ec="#aaaacc", lw=0.7))
ax4.set_xlim(lim_min, lim_max)
ax4.set_ylim(lim_min, lim_max)
ax4.set_title("(d)  Calculated vs. Experimental Wavenumber Correlation",
              fontweight="bold", loc="left", pad=5)
ax4.legend(loc="lower right", frameon=True, edgecolor="#cccccc", fontsize=7.5)


# Panel e — Residuals
style_ax(ax5,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="Residual (cm$^{-1}$)")
ax5.scatter(theo_v, residuals, s=sizes, c=theo_int_match,
            cmap="YlOrRd", zorder=5, edgecolors="#333333", lw=0.4, alpha=0.88)
ax5.axhline(0,     color=C_ZERO,    lw=1.0, ls="--", zorder=3)
ax5.axhline( MAE,  color="#2ecc71", lw=0.9, ls=":",  label=f"\u00b1MAE = {MAE:.1f} cm\u207b\u00b9")
ax5.axhline(-MAE,  color="#2ecc71", lw=0.9, ls=":")
ax5.axhline( RMSE, color="#e74c3c", lw=0.9, ls="-.", label=f"\u00b1RMSE = {RMSE:.1f} cm\u207b\u00b9")
ax5.axhline(-RMSE, color="#e74c3c", lw=0.9, ls="-.")
ax5.set_title("(e)  Residuals: \u03bd$_{exp}$ \u2212 SF\u00b7\u03bd$_{calc}$",
              fontweight="bold", loc="left", pad=5)
ax5.legend(loc="upper right", frameon=True, edgecolor="#cccccc", fontsize=7.5)
rlim = max(np.abs(residuals).max() * 1.3, RMSE * 2)
ax5.set_ylim(-rlim, rlim)
ax5.set_xlim(lim_min, lim_max)


# Supertitle & footnote
fig.suptitle(
    f"{MOLECULE}: Validation of DFT/B3LYP/6-311++G(d,p) Calculated Raman Spectrum\n"
    "against Experimental Raman Data \u2014 Vibrational Frequency Scaling Factor Analysis",
    fontsize=12, fontweight="bold", y=0.998)

fig.text(0.5, 0.008,
         f"Peak matching: automatic, \u00b1{TOLERANCE} cm\u207b\u00b9 tolerance, "
         f"{n_matched} pairs ({pct_theo:.0f}% of {n_theo_total} "
         f"calculated modes in 400\u20133300 cm\u207b\u00b9).  "
         "SG smoothing applied to peak detection only; raw data plotted.  "
         "Lorentzian broadening (FWHM = 10 cm\u207b\u00b9) for visual comparison only.",
         ha="center", fontsize=7, color="#666666", style="italic")

plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved -> {OUTPUT}")




# %%
"""
D-Tagatose: DFT/B3LYP/6-311++G(d,p) Raman Vibrational Frequency Scaling Factor Analysis
==========================================================================================

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

# ─── FILE PATHS ────────────────────────────────────────────────────────────────
THEO_FILE = "d-tagatose-raman.txt"
EXP_FILE  = "TAG 50 x 532nm 2400g 50p.txt"
OUTPUT    = "d-tagatose_raman.png"
MOLECULE  = "D-Tagatose"
# ──────────────────────────────────────────────────────────────────────────────


# STEP 1 — LOAD & SORT
theo_raw = np.loadtxt(THEO_FILE)
exp_raw  = np.loadtxt(EXP_FILE)
theo_raw = theo_raw[np.argsort(theo_raw[:, 0])]
exp_raw  = exp_raw [np.argsort(exp_raw [:, 0])]

exp_wn   = exp_raw[:, 0]
exp_y    = exp_raw[:, 1]
exp_norm = exp_y / exp_y.max()


# STEP 2 — THEORETICAL PEAKS (400-3300 cm-1)
mask_theo    = (theo_raw[:, 0] >= 400) & (theo_raw[:, 0] <= 3300)
theo_wn_all  = theo_raw[mask_theo, 0]
theo_int_all = theo_raw[mask_theo, 1]  
theo_norm    = theo_int_all / theo_int_all.max()
n_theo_total = len(theo_wn_all)

print(f"Theoretical peaks (400-3300 cm-1): n = {n_theo_total}")


# STEP 3 — SAVITZKY-GOLAY 
SG_WINDOW = 21
SG_POLY   = 3
sg_smooth = savgol_filter(exp_y, window_length=SG_WINDOW, polyorder=SG_POLY)

noise_std = np.std(exp_y[(exp_wn > 1800) & (exp_wn < 2200)])
print(f"Noise estimate (1800-2200 cm-1 region): {noise_std:.1f} counts")


# STEP 4 — EXPERIMENTAL PEAK DETECTION
PROMINENCE_FRAC = 0.03
PROMINENCE      = PROMINENCE_FRAC * exp_y.max()
DISTANCE        = 25

exp_pk_idx, _ = find_peaks(sg_smooth, prominence=PROMINENCE, distance=DISTANCE)
exp_pk_wn     = exp_wn[exp_pk_idx]
exp_pk_y      = exp_y [exp_pk_idx]
n_exp_detected = len(exp_pk_idx)

print(f"Experimental peaks detected: n = {n_exp_detected}")
print(f"  SG: window={SG_WINDOW}, polyorder={SG_POLY}")
print(f"  find_peaks: prominence >= {PROMINENCE_FRAC*100:.0f}% of max "
      f"({PROMINENCE:.0f} counts), distance >= {DISTANCE} pts")
print()


# STEP 5 — AUTOMATIC PEAK MATCHING
TOLERANCE = 30

priority_order = np.argsort(-theo_int_all)
used_exp_idx   = set()
matched_pairs  = []

for idx in priority_order:
    tw = theo_wn_all[idx]
    ti = theo_int_all[idx]
    diffs = np.abs(exp_pk_wn - tw)
    j = int(np.argmin(diffs))
    if diffs[j] <= TOLERANCE and j not in used_exp_idx:
        matched_pairs.append({
            "theo_wn"  : tw,
            "exp_wn"   : exp_pk_wn[j],
            "theo_int" : ti,
            "exp_y"    : exp_pk_y[j],
            "delta"    : exp_pk_wn[j] - tw,
            "sf"       : exp_pk_wn[j] / tw,
        })
        used_exp_idx.add(j)

matched_pairs.sort(key=lambda x: x["theo_wn"])

theo_v         = np.array([p["theo_wn"]  for p in matched_pairs])
exp_v          = np.array([p["exp_wn"]   for p in matched_pairs])
sfs            = np.array([p["sf"]       for p in matched_pairs])
theo_int_match = np.array([p["theo_int"] for p in matched_pairs])
n_matched      = len(matched_pairs)

pct_theo = 100 * n_matched / n_theo_total
pct_exp  = 100 * n_matched / n_exp_detected


# STEP 6 — SCALING FACTOR STATISTICS
SF_mean   = sfs.mean()
SF_std    = sfs.std(ddof=1)
SF_median = np.median(sfs)
n_pairs   = len(sfs)
SF_95lo   = SF_mean - 1.96 * SF_std / np.sqrt(n_pairs)
SF_95hi   = SF_mean + 1.96 * SF_std / np.sqrt(n_pairs)

SF        = np.sum(theo_v * exp_v) / np.sum(theo_v**2)
residuals = exp_v - SF * theo_v
SS_res    = np.sum(residuals**2)
SS_tot    = np.sum((exp_v - exp_v.mean())**2)
R2        = 1 - SS_res / SS_tot
MAE       = np.mean(np.abs(residuals))
RMSE      = np.sqrt(np.mean(residuals**2))

slope_f, intercept_f, r_f, _, _ = linregress(theo_v, exp_v)

print("=" * 65)
print(f"TABLE S — Automatic peak matching results ({MOLECULE}, Raman)")
print(f"  Wavenumber range            : 400-3300 cm-1")
print(f"  Theoretical peaks in range  : {n_theo_total}")
print(f"  Experimental peaks detected : {n_exp_detected}")
print(f"  Matched pairs               : {n_matched} "
      f"({pct_theo:.0f}% of theoretical, {pct_exp:.0f}% of experimental)")
print(f"  Tolerance                   : +/-{TOLERANCE} cm-1")
print("=" * 65)
print(f"{'nu_calc (cm-1)':>16}  {'nu_exp (cm-1)':>14}  {'Delta_nu':>10}  {'SF_i':>10}")
print("-" * 58)
for p in matched_pairs:
    print(f"{p['theo_wn']:>16.1f}  {p['exp_wn']:>14.1f}  "
          f"{p['delta']:>+10.1f}  {p['sf']:>10.5f}")
print("-" * 58)
print(f"\nSCALING FACTOR SUMMARY")
print(f"  Method A mean ratio   : {SF_mean:.5f}   SD = {SF_std:.5f}")
print(f"  95% CI (mean)         : [{SF_95lo:.4f}, {SF_95hi:.4f}]")
print(f"  Median SF             : {SF_median:.5f}")
print(f"  Method B regression   : {SF:.5f}")
print(f"  R2                    : {R2:.6f}")
print(f"  MAE                   : {MAE:.2f} cm-1")
print(f"  RMSE                  : {RMSE:.2f} cm-1")
print()


# STEP 7 — LORENTZIAN BROADENING
def lorentzian(wn_grid, centers, intensities, fwhm=10.0):
    spec = np.zeros_like(wn_grid, dtype=float)
    for c, a in zip(centers, intensities):
        spec += a / (1.0 + ((wn_grid - c) / (fwhm / 2.0))**2)
    return spec / spec.max() if spec.max() > 0 else spec

wn_grid    = np.linspace(400, 3300, 5000)
theo_broad = lorentzian(wn_grid, theo_wn_all, theo_norm, fwhm=10)


# STEP 8 — FIVE-PANEL FIGURE 
matplotlib.rcParams.update({
    "font.family"     : "sans-serif",
    "font.size"       : 10,
    "axes.labelsize"  : 11,
    "axes.titlesize"  : 10.5,
    "xtick.labelsize" : 9.5,
    "ytick.labelsize" : 9.5,
    "legend.fontsize" : 9,
    "axes.labelweight": "bold",
    "figure.dpi"      : 300,
})

C_EXP   = "#1a6faf"
C_THEO  = "#c0392b"
C_MATCH = "#d4820a"
C_GRID  = "#efefef"
C_ZERO  = "#aaaaaa"

def style_ax(ax, xlabel="Raman shift (cm\u207b\u00b9)", ylabel=None):
    ax.set_facecolor("white")
    ax.tick_params(which="major", direction="out", length=4, width=0.9,
                   color="#444444", labelcolor="#111111")
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="out", length=2, width=0.5,
                   color="#999999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888");   ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_color("#888888"); ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(True, which="major", color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)

fig = plt.figure(figsize=(14, 18), facecolor="white")
gs_outer = gridspec.GridSpec(
    4, 1, figure=fig, hspace=0.54,
    top=0.945, bottom=0.048, left=0.09, right=0.965,
    height_ratios=[1, 1, 1, 1.05]
)
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_outer[3], wspace=0.38, width_ratios=[1.5, 1]
)
ax1 = fig.add_subplot(gs_outer[0])
ax2 = fig.add_subplot(gs_outer[1])
ax3 = fig.add_subplot(gs_outer[2])
ax4 = fig.add_subplot(gs_bottom[0])
ax5 = fig.add_subplot(gs_bottom[1])

XLO, XHI = 400, 3300


# Panel a — Experimental Raman
style_ax(ax1, ylabel="Normalized intensity")
ax1.plot(exp_wn, exp_norm, color=C_EXP, lw=1.0,
         label="Experimental Raman spectrum (532 nm, 2400 g/mm, normalized)")
for w, y in zip(exp_pk_wn, exp_pk_y / exp_y.max()):
    if y > 0.06:
        ax1.text(w, y + 0.04, f"{w:.0f}", fontsize=6, color=C_EXP,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax1.set_xlim(XLO, XHI)
ax1.set_ylim(-0.04, 1.50)
ax1.set_title(f"(a)  Experimental Raman Spectrum \u2014 {MOLECULE}",
              fontweight="bold", loc="left", pad=5)
ax1.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax1.text(0.005, 0.97,
         f"SG smoothing (w = {SG_WINDOW}, p = {SG_POLY}) for peak detection only  |  "
         f"{n_exp_detected} peaks detected "
         f"(prominence \u2265 {PROMINENCE_FRAC*100:.0f}% of max, distance \u2265 {DISTANCE} pts)  |  "
         "Raw data plotted.",
         transform=ax1.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel b — DFT Raman stick spectrum
style_ax(ax2, xlabel="Raman shift (cm\u207b\u00b9)", ylabel="Normalized Raman intensity")
ax2.vlines(theo_wn_all, 0, theo_norm, color=C_THEO, lw=1.5, alpha=0.85,
           label="DFT/B3LYP/6-311++G(d,p) calculated Raman spectrum (unscaled)")
for w, i in zip(theo_wn_all, theo_norm):
    if i > 0.15:
        ax2.text(w, i + 0.03, f"{w:.0f}", fontsize=6, color=C_THEO,
                 ha="center", rotation=90, va="bottom", clip_on=True, fontweight="bold")
ax2.set_xlim(XLO, XHI)
ax2.set_ylim(-0.04, 1.50)
ax2.set_title(f"(b)  Calculated Raman Spectrum \u2014 DFT/B3LYP/6-311++G(d,p)",
              fontweight="bold", loc="left", pad=5)
ax2.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax2.text(0.005, 0.97,
         f"Gaussian frequency output  |  {n_theo_total} modes in 400\u20133300 cm\u207b\u00b9  |  "
         "Unscaled  |  Y-axis: Raman intensity (normalized)",
         transform=ax2.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel c — Overlay + matched peaks
style_ax(ax3, xlabel="Raman shift (cm\u207b\u00b9)", ylabel="Normalized intensity")
ax3.plot(exp_wn, exp_norm, color=C_EXP, lw=1.0, alpha=0.92,
         label="Experimental (normalized intensity)")
ax3.plot(wn_grid, theo_broad * 0.80, color=C_THEO, lw=0.9, alpha=0.65,
         label="Calculated (Lorentzian-broadened, FWHM = 10 cm\u207b\u00b9, for visual comparison only)")
for p in matched_pairs:
    idx_e = int(np.argmin(np.abs(exp_wn - p["exp_wn"])))
    y_e   = exp_norm[idx_e]
    ew, tw = p["exp_wn"], p["theo_wn"]
    ax3.plot([ew, ew],  [y_e, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot([ew, tw],  [-0.04, -0.04], color=C_MATCH, lw=0.7, alpha=0.45)
    ax3.plot(ew,  y_e,  "o", color=C_EXP,  ms=4,   zorder=6, mew=0.6, mec="white")
    ax3.plot(tw, -0.04, "s", color=C_THEO, ms=3.5, zorder=6, mew=0.6, mec="white")
ax3.axhline(-0.04, color="#cccccc", lw=0.6, ls=":")
ax3.text(XHI - 30, -0.04, "calc.", color=C_THEO, fontsize=7,
         va="center", ha="right", fontweight="bold")
ax3.set_xlim(XLO, XHI)
ax3.set_ylim(-0.10, 1.30)
ax3.set_title(
    f"(c)  Spectral Overlay and Peak Correspondence  |  "
    f"n = {n_matched} matched pairs  |  "
    f"400\u20133300 cm\u207b\u00b9  |  "
    f"{pct_theo:.0f}% of calculated modes matched",
    fontweight="bold", loc="left", pad=5)
ax3.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
ax3.text(0.005, 0.97,
         f"Matching: nearest-neighbor within \u00b1{TOLERANCE} cm\u207b\u00b9; "
         "priority by descending calculated Raman intensity; each experimental peak used once.\n"
         "Lorentzian broadening for visual comparison only \u2014 "
         "scaling factor derived from discrete peak positions.",
         transform=ax3.transAxes, fontsize=7, color="#333333", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#cccccc", lw=0.7))


# Panel d — Correlation
style_ax(ax4,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="\u03bd$_{exp}$ (cm$^{-1}$)")
sizes = 30 + 140 * (theo_int_match / theo_int_match.max())
sc = ax4.scatter(theo_v, exp_v, s=sizes, c=theo_int_match,
                 cmap="YlOrRd", zorder=5, edgecolors="#222222",
                 lw=0.5, alpha=0.88,
                 label=f"Matched peaks (n = {n_matched}, size/color \u221d Raman intensity)")
lim_min, lim_max = 380, 3150
x_fit = np.linspace(lim_min, lim_max, 500)
ax4.plot(x_fit, x_fit, "--", color="#888888", lw=1.2, label="1:1 reference")
ax4.plot(x_fit, SF * x_fit, "-", color=C_THEO, lw=2.0,
         label=f"Origin-forced: slope = {SF:.4f}")
cb = plt.colorbar(sc, ax=ax4, pad=0.01, shrink=0.8)
cb.set_label("Raman intensity", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
stats = (f"SF = {SF:.5f}  (origin regression)\n"
         f"R\u00b2 = {R2:.6f}\n"
         f"MAE = {MAE:.2f} cm\u207b\u00b9   RMSE = {RMSE:.2f} cm\u207b\u00b9\n"
         f"Mean SF = {SF_mean:.5f}   SD = {SF_std:.5f}\n"
         f"95% CI [{SF_95lo:.4f}, {SF_95hi:.4f}]")
ax4.text(0.03, 0.97, stats, transform=ax4.transAxes,
         fontsize=8.5, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f0f4ff", ec="#9999cc", lw=0.8))
ax4.set_xlim(lim_min, lim_max)
ax4.set_ylim(lim_min, lim_max)
ax4.set_title("(d)  Calculated vs. Experimental Wavenumber Correlation",
              fontweight="bold", loc="left", pad=5)
ax4.legend(loc="lower right", frameon=True, edgecolor="#bbbbbb")


# Panel e — Residuals
style_ax(ax5,
         xlabel="\u03bd$_{calc}$ (cm$^{-1}$)",
         ylabel="Residual (cm$^{-1}$)")
ax5.scatter(theo_v, residuals, s=sizes, c=theo_int_match,
            cmap="YlOrRd", zorder=5, edgecolors="#222222", lw=0.5, alpha=0.88)
ax5.axhline(0,     color=C_ZERO,    lw=1.1, ls="--", zorder=3)
ax5.axhline( MAE,  color="#2ecc71", lw=1.0, ls=":",  label=f"\u00b1MAE = {MAE:.1f} cm\u207b\u00b9")
ax5.axhline(-MAE,  color="#2ecc71", lw=1.0, ls=":")
ax5.axhline( RMSE, color="#e74c3c", lw=1.0, ls="-.", label=f"\u00b1RMSE = {RMSE:.1f} cm\u207b\u00b9")
ax5.axhline(-RMSE, color="#e74c3c", lw=1.0, ls="-.")
ax5.set_title("(e)  Residuals: \u03bd$_{exp}$ \u2212 SF\u00b7\u03bd$_{calc}$",
              fontweight="bold", loc="left", pad=5)
ax5.legend(loc="upper right", frameon=True, edgecolor="#bbbbbb")
rlim = max(np.abs(residuals).max() * 1.3, RMSE * 2)
ax5.set_ylim(-rlim, rlim)
ax5.set_xlim(lim_min, lim_max)


# Supertitle & footnote
fig.suptitle(
    f"{MOLECULE}: Validation of DFT/B3LYP/6-311++G(d,p) Calculated Raman Spectrum\n"
    "against Experimental Raman Data \u2014 Vibrational Frequency Scaling Factor Analysis",
    fontsize=13, fontweight="bold", y=0.998)

fig.text(0.5, 0.008,
         f"Peak matching: automatic, \u00b1{TOLERANCE} cm\u207b\u00b9 tolerance, "
         f"{n_matched} pairs ({pct_theo:.0f}% of {n_theo_total} "
         "calculated modes, 400\u20133300 cm\u207b\u00b9).  "
         "SG smoothing applied to peak detection only; raw data plotted.  "
         "Lorentzian broadening (FWHM = 10 cm\u207b\u00b9) for visual comparison only.",
         ha="center", fontsize=7.5, color="#444444", style="italic")

plt.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Figure saved -> {OUTPUT}")

