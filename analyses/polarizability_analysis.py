# %%
import os
import re
import pandas as pd
import numpy as np
from cclib.io import ccopen

# Paths
folder_path = "C:/Users/pc/Desktop/sweeteners_database"
sweetness_file_path = "C:/Users/pc/Desktop/ChemTastesDB/foods-2444588 - Table S2.xlsx"
output_path = "C:/Users/pc/Desktop/sweeteners_polarizability_dataset.csv"
error_log_path = "C:/Users/pc/Desktop/sweeteners_polarizability_error_log.csv"
regex_pattern = r"^\d+_[\w-]+\.inp\.out_\d+$"

# Extra compounds
compounds_extra = [
    {"Order": 181, "Molecule": "dulcoside-A", "FilePath": r"C:\Users\pc\Desktop\results\results\dulcoside-a.inp.out_173347"},
    {"Order": 182, "Molecule": "rebaudioside-C", "FilePath": r"C:\Users\pc\Desktop\results\results\reba-c.inp.out_173343"},
    {"Order": 248, "Molecule": "rebaudioside-E", "FilePath": r"C:\Users\pc\Desktop\results\results\reba-e.inp.out_173340"},
    {"Order": 254, "Molecule": "rebaudioside-A", "FilePath": r"C:\Users\pc\Desktop\results\results\reba-a.inp.out_173337"},
    {"Order": 259, "Molecule": "rebaudioside-B", "FilePath": r"C:\Users\pc\Desktop\results\results\reba-b.inp.out_173339"},
    {"Order": 260, "Molecule": "rebaudioside-D", "FilePath": r"C:\Users\pc\Desktop\results\results\reba-d.inp.out_173342"}
]

def extract_molecule_info(file_name):
    match = re.match(r"^(\d+)_([\w-]+)", file_name)
    return (int(match.group(1)), match.group(2)) if match else (None, None)

def read_polarizability(file_path):
    try:
        data = ccopen(file_path).parse()
        return np.array(data.polarizabilities[-1]) if hasattr(data, "polarizabilities") else None
    except Exception as e:
        return str(e)

def compute_anisotropy(tensor):
    """Rotation-invariant anisotropy from eigenvalues"""
    if tensor is None:
        return None
    eigenvalues = np.linalg.eigvalsh(tensor)
    a1, a2, a3 = np.sort(eigenvalues)
    return np.sqrt(0.5 * ((a1-a2)**2 + (a2-a3)**2 + (a3-a1)**2))

def compute_trace(tensor):
    """Trace of polarizability tensor"""
    return None if tensor is None else np.trace(tensor)

def compute_determinant(tensor):
    """Determinant of polarizability tensor"""
    return None if tensor is None else np.linalg.det(tensor)

def compute_eigenvalues(tensor):
    """Eigenvalues and range"""
    if tensor is None:
        return None, None, None, None
    eigenvalues = np.linalg.eigvalsh(tensor)
    eig_sorted = np.sort(eigenvalues)
    return eig_sorted[0], eig_sorted[1], eig_sorted[2], eig_sorted[2] - eig_sorted[0]

# Process main folder
data = []
error_log = []

for file in sorted(
        [file for file in os.listdir(folder_path) if re.match(regex_pattern, file)],
        key=lambda x: int(re.match(r"^(\d+)", x).group())):
    file_path = os.path.join(folder_path, file)
    order, molecule = extract_molecule_info(file)
    tensor = read_polarizability(file_path)

    if isinstance(tensor, str):
        error_log.append({"Molecule": f"{order}-{molecule}", "Error": tensor})
        continue

    if tensor is not None:
        eig1, eig2, eig3, eigen_range = compute_eigenvalues(tensor)
        data.append({
            "Molecule": f"{order}-{molecule}",
            "Order": order,
            "XX": tensor[0][0],
            "XY": tensor[0][1],
            "XZ": tensor[0][2],
            "YX": tensor[1][0],
            "YY": tensor[1][1],
            "YZ": tensor[1][2],
            "ZX": tensor[2][0],
            "ZY": tensor[2][1],
            "ZZ": tensor[2][2],
            "XX-YY": tensor[0][0] - tensor[1][1],
            "YY-ZZ": tensor[1][1] - tensor[2][2],
            "XX-ZZ": tensor[0][0] - tensor[2][2],
            "Anisotropy": compute_anisotropy(tensor),
            "Trace": compute_trace(tensor),
            "Determinant": compute_determinant(tensor),
            "Eigenvalue1": eig1,
            "Eigenvalue2": eig2,
            "Eigenvalue3": eig3,
            "EigenRange": eigen_range
        })
    else:
        error_log.append({"Molecule": f"{order}-{molecule}", "Error": "Polarizability not found"})

# Process extra compounds
for comp in compounds_extra:
    order, molecule, file_path = comp["Order"], comp["Molecule"], comp["FilePath"]
    tensor = read_polarizability(file_path)

    if isinstance(tensor, str):
        error_log.append({"Molecule": f"{order}-{molecule}", "Error": tensor})
        continue

    if tensor is not None:
        eig1, eig2, eig3, eigen_range = compute_eigenvalues(tensor)
        data.append({
            "Molecule": f"{order}-{molecule}",
            "Order": order,
            "XX": tensor[0][0],
            "XY": tensor[0][1],
            "XZ": tensor[0][2],
            "YX": tensor[1][0],
            "YY": tensor[1][1],
            "YZ": tensor[1][2],
            "ZX": tensor[2][0],
            "ZY": tensor[2][1],
            "ZZ": tensor[2][2],
            "XX-YY": tensor[0][0] - tensor[1][1],
            "YY-ZZ": tensor[1][1] - tensor[2][2],
            "XX-ZZ": tensor[0][0] - tensor[2][2],
            "Anisotropy": compute_anisotropy(tensor),
            "Trace": compute_trace(tensor),
            "Determinant": compute_determinant(tensor),
            "Eigenvalue1": eig1,
            "Eigenvalue2": eig2,
            "Eigenvalue3": eig3,
            "EigenRange": eigen_range
        })
    else:
        error_log.append({"Molecule": f"{order}-{molecule}", "Error": "Polarizability not found"})

# Create DataFrame
df_polarizability = pd.DataFrame(data)
df_polarizability.sort_values(by="Order", inplace=True)

# Tensor symmetry validation
print("\n=== TENSOR SYMMETRY VALIDATION ===")
xy_diff = abs(df_polarizability['XY'] - df_polarizability['YX']).max()
xz_diff = abs(df_polarizability['XZ'] - df_polarizability['ZX']).max()
yz_diff = abs(df_polarizability['YZ'] - df_polarizability['ZY']).max()

print(f"Max |XY - YX|: {xy_diff:.10f}")
print(f"Max |XZ - ZX|: {xz_diff:.10f}")
print(f"Max |YZ - ZY|: {yz_diff:.10f}")

if max(xy_diff, xz_diff, yz_diff) > 0.01:
    print("WARNING: Tensor symmetry violation detected")
else:
    print("Tensor symmetry verified")

# Off-diagonal magnitude check
print("\n=== OFF-DIAGONAL MAGNITUDE CHECK ===")
max_xy = df_polarizability['XY'].abs().max()
max_xz = df_polarizability['XZ'].abs().max()
max_yz = df_polarizability['YZ'].abs().max()
mean_diag = (df_polarizability['XX'].abs().mean() + 
             df_polarizability['YY'].abs().mean() + 
             df_polarizability['ZZ'].abs().mean()) / 3

print(f"Max |XY|: {max_xy:.6f}")
print(f"Max |XZ|: {max_xz:.6f}")
print(f"Max |YZ|: {max_yz:.6f}")
print(f"Mean diagonal: {mean_diag:.2f}")
print(f"Off-diagonal/Diagonal ratio: {max(max_xy, max_xz, max_yz)/mean_diag:.6f}")

# Merge with sweetness data
sweetness_data = pd.read_excel(sweetness_file_path, sheet_name="Table S2")
df_polarizability = df_polarizability.merge(
    sweetness_data[["ID", "logSw"]], 
    left_on="Order", 
    right_on="ID", 
    how="left"
)
df_polarizability.drop(columns=["ID"], inplace=True)
df_polarizability.rename(columns={"logSw": "Sweetness (logSw)"}, inplace=True)

# Save
df_polarizability.to_csv(output_path, index=False)
pd.DataFrame(error_log).to_csv(error_log_path, index=False)

print(f"\n=== RESULTS ===")
print(f"Total compounds: {df_polarizability.shape[0]}")
print(f"Data saved to: {output_path}")
print(f"Error log saved to: {error_log_path}")


# %% 
import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('sweeteners_polarizability_dataset.csv')

print("=== DATA VALIDATION ===")
print(f"Dataset shape: {df.shape}")

# Check tensor symmetry
xy_diff = abs(df['XY'] - df['YX']).max()
xz_diff = abs(df['XZ'] - df['ZX']).max()
yz_diff = abs(df['YZ'] - df['ZY']).max()

print(f"Tensor symmetry check:")
print(f" Max |XY - YX|: {xy_diff:.6f}")
print(f" Max |XZ - ZX|: {xz_diff:.6f}")
print(f" Max |YZ - ZY|: {yz_diff:.6f}")

if max(xy_diff, xz_diff, yz_diff) > 0.01:
    print("WARNING: Tensor symmetry violation detected!")
else:
    print("Tensor symmetry verified")

# Check for missing values
missing_count = df.isnull().sum().sum()
print(f"Missing values: {missing_count}")

print("\n" + "="*60)
print("=== STATISTICAL ANALYSIS ===\n")

# 1. CONTINUOUS VARIABLE CORRELATION ANALYSIS
print("1. SPEARMAN CORRELATION ANALYSIS")
print("-" * 50)

polarizability_features = [
    "Anisotropy", "Trace", "Determinant", 
    "Eigenvalue1", "Eigenvalue2", "Eigenvalue3", "EigenRange"
]

# Spearman correlations
correlations = []
for feature in polarizability_features:
    corr, p_val = spearmanr(df[feature], df["Sweetness (logSw)"])
    correlations.append({
        'Feature': feature,
        'Spearman_rho': corr,
        'p_value': p_val,
        'abs_rho': abs(corr)
    })

# Sort by absolute correlation
corr_df = pd.DataFrame(correlations).sort_values('abs_rho', ascending=False)

# FDR correction
_, p_adjusted, _, _ = multipletests(corr_df['p_value'], method='fdr_bh')
corr_df['p_adjusted'] = p_adjusted
corr_df['significant'] = p_adjusted < 0.05

print(f"{'Feature':<15} {'Spearman ρ':<12} {'p-value':<10} {'p_adj':<10} {'Sig':<5}")
print("-" * 60)
for _, row in corr_df.iterrows():
    sig = "***" if row['p_adjusted'] < 0.001 else "**" if row['p_adjusted'] < 0.01 else "*" if row['p_adjusted'] < 0.05 else "ns"
    print(f"{row['Feature']:<15} {row['Spearman_rho']:>11.3f} {row['p_value']:>9.4f} {row['p_adjusted']:>9.4f} {sig:<5}")

# 2. CATEGORICAL GROUPING
print(f"\n2. CATEGORICAL GROUPING")
print("-" * 50)

sweetness_stats = df["Sweetness (logSw)"].describe()
print("Sweetness distribution:")
print(sweetness_stats)

# Tertile thresholds
low_thr = df["Sweetness (logSw)"].quantile(1/3)
high_thr = df["Sweetness (logSw)"].quantile(2/3)

print(f"\nTertile thresholds:")
print(f"Low-Medium: {low_thr:.3f}")
print(f"Medium-High: {high_thr:.3f}")

# Categorical grouping
df["Sweetness Category"] = pd.cut(
    df["Sweetness (logSw)"],
    bins=[df["Sweetness (logSw)"].min() - 0.001, low_thr, high_thr, df["Sweetness (logSw)"].max() + 0.001],
    labels=["Low", "Medium", "High"]
)

print(f"\nGroup sizes:")
print(df["Sweetness Category"].value_counts().sort_index())

# 3. GROUP COMPARISON ANALYSIS
print(f"\n3. GROUP COMPARISON ANALYSIS")
print("=" * 60)

def cliff_delta(x, y):
    """Cliff's Delta effect size calculation"""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    delta = 0
    for i in x:
        for j in y:
            if i > j:
                delta += 1
            elif i < j:
                delta -= 1
    return delta / (n1 * n2)

def interpret_cliff_delta(delta):
    """Interpret Cliff's Delta effect size"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"

# Detailed analysis results
detailed_results = []

for feature in polarizability_features:
    print(f"\n{'='*20} {feature} {'='*20}")
    
    groups_data = {}
    group_stats = {}
    
    for group in ["Low", "Medium", "High"]:
        data = df[df["Sweetness Category"] == group][feature].dropna()
        groups_data[group] = data.values
        group_stats[group] = {
            'n': len(data),
            'median': data.median(),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'mean': data.mean(),
            'std': data.std()
        }
    
    # Descriptive statistics
    print("Descriptive Statistics:")
    print(f"{'Group':<8} {'n':<4} {'Median':<8} {'IQR':<8} {'Mean':<8} {'SD':<8}")
    print("-" * 50)
    for group in ["Low", "Medium", "High"]:
        stats = group_stats[group]
        print(f"{group:<8} {stats['n']:<4} {stats['median']:<8.2f} {stats['iqr']:<8.2f} {stats['mean']:<8.2f} {stats['std']:<8.2f}")
    
    # Omnibus test
    if all(len(groups_data[g]) >= 1 for g in ["Low", "Medium", "High"]):
        h_stat, p_omnibus = kruskal(*[groups_data[g] for g in ["Low", "Medium", "High"]])
        
        # Corrected eta-squared calculation
        n_total = sum(len(groups_data[g]) for g in ["Low", "Medium", "High"])
        k = 3 
        eta_squared = (h_stat - k + 1) / (n_total - k) if n_total > k else 0
        
        print(f"\nKruskal-Wallis Test:")
        print(f" H = {h_stat:.3f}, p = {p_omnibus:.4f}")
        print(f" η² = {eta_squared:.3f} (effect size)")
        
        result_summary = {
            'Feature': feature,
            'H_statistic': h_stat,
            'p_omnibus': p_omnibus,
            'eta_squared': eta_squared,
            'significant': p_omnibus < 0.05,
            'post_hoc_results': []
        }
        
        # Post-hoc tests only if omnibus is significant
        if p_omnibus < 0.05:
            print(f"\nPost-hoc Analysis (Mann-Whitney U):")
            pairs = [("Low", "Medium"), ("Low", "High"), ("Medium", "High")]
            p_values = []
            
            for g1, g2 in pairs:
                if len(groups_data[g1]) > 0 and len(groups_data[g2]) > 0:
                    stat, p = mannwhitneyu(groups_data[g1], groups_data[g2], alternative="two-sided")
                    
                    # Effect size (Cliff's Delta)
                    cliff_delta_val = cliff_delta(groups_data[g1], groups_data[g2])
                    effect_interp = interpret_cliff_delta(cliff_delta_val)
                    
                    p_values.append(p)
                    result_summary['post_hoc_results'].append({
                        'comparison': f"{g1} vs {g2}",
                        'U_statistic': stat,
                        'p_value': p,
                        'cliff_delta': cliff_delta_val,
                        'effect_size': effect_interp
                    })
                    
                    print(f" {g1} vs {g2}: U={stat:.1f}, p={p:.4f}, δ={cliff_delta_val:.3f} ({effect_interp})")
            
            # FDR correction
            if p_values:
                _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
                print(f"\nFDR Correction:")
                for i, (g1, g2) in enumerate(pairs):
                    if i < len(p_adjusted):
                        sig = "***" if p_adjusted[i] < 0.001 else "**" if p_adjusted[i] < 0.01 else "*" if p_adjusted[i] < 0.05 else "ns"
                        result_summary['post_hoc_results'][i]['p_adjusted'] = p_adjusted[i]
                        result_summary['post_hoc_results'][i]['significant_adj'] = p_adjusted[i] < 0.05
                        print(f" {g1} vs {g2}: p_adj={p_adjusted[i]:.4f} {sig}")
        else:
            print(" Omnibus test not significant - no post-hoc tests performed")
        
        detailed_results.append(result_summary)

# 4. SUMMARY RESULTS
print(f"\n{'='*60}")
print("4. SUMMARY RESULTS")
print(f"{'='*60}")

# Significant features
significant_features = [r for r in detailed_results if r['significant']]
print(f"\nSignificant group differences ({len(significant_features)}/{len(polarizability_features)}):")

# Effect size ranking
for result in sorted(significant_features, key=lambda x: x['eta_squared'], reverse=True):
    print(f" {result['Feature']:<15}: η²={result['eta_squared']:.3f}, p={result['p_omnibus']:.4f}")

# Strongest post-hoc results
print(f"\nStrongest Group Comparisons (large effect size):")
large_effects = []
for result in significant_features:
    for comparison in result['post_hoc_results']:
        if interpret_cliff_delta(comparison['cliff_delta']) == "large":
            large_effects.append({
                'feature': result['Feature'],
                'comparison': comparison['comparison'],
                'cliff_delta': comparison['cliff_delta'],
                'p_adjusted': comparison.get('p_adjusted', comparison['p_value'])
            })

for effect in sorted(large_effects, key=lambda x: abs(x['cliff_delta']), reverse=True)[:10]:
    print(f" {effect['feature']:<15} ({effect['comparison']:<15}): δ={effect['cliff_delta']:>6.3f}, p={effect['p_adjusted']:.4f}")

# 5. METHODOLOGICAL NOTES
print(f"\n5. METHODOLOGICAL NOTES")
print("-" * 50)
print("✓ Non-parametric tests used (data not normally distributed)")
print("✓ Multiple comparison correction applied (FDR)")
print("✓ Effect sizes reported (Cliff's Delta)")
print("✓ Omnibus test priority maintained")
print("✓ Continuous variable correlations analyzed")
print("✓ Data validation performed (symmetry, missing values)")

print(f"\nAnalysis completed successfully.")

# Save correlation results
corr_df.to_csv('spearman_polarizability_invariant.csv', index=False)
print(f"\nCorrelation results saved to: spearman_polarizability_invariant.csv")

# Save group comparison results
group_results_data = []
for result in detailed_results:
    if result['post_hoc_results']:
        for comparison in result['post_hoc_results']:
            group_results_data.append({
                'Feature': result['Feature'],
                'H_statistic': result['H_statistic'],
                'p_omnibus': result['p_omnibus'],
                'eta_squared': result['eta_squared'],
                'comparison': comparison['comparison'],
                'U_statistic': comparison['U_statistic'],
                'p_value': comparison['p_value'],
                'p_adjusted': comparison.get('p_adjusted', np.nan),
                'cliff_delta': comparison['cliff_delta'],
                'cliff_delta_interpretation': comparison['effect_size'],
                'significant_adj': comparison.get('significant_adj', False)
            })
    else:
        group_results_data.append({
            'Feature': result['Feature'],
            'H_statistic': result['H_statistic'],
            'p_omnibus': result['p_omnibus'],
            'eta_squared': result['eta_squared'],
            'comparison': np.nan,
            'U_statistic': np.nan,
            'p_value': np.nan,
            'p_adjusted': np.nan,
            'cliff_delta': np.nan,
            'cliff_delta_interpretation': np.nan,
            'significant_adj': np.nan
        })

group_df = pd.DataFrame(group_results_data)
group_df.to_csv('group_tests_polarizability_invariant.csv', index=False)
print(f"Group comparison results saved to: group_tests_polarizability_invariant.csv")

