import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

INFILE = "uci_credit_default.csv"
OUTDIR = "eda_outputs"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INFILE)
print("Loaded file:", INFILE)
print("Shape:", df.shape)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values (per column) ---")
print(df.isnull().sum())

if 'default' not in df.columns:
    raise SystemExit("ERROR: target column 'default' not found.")
vc = df['default'].value_counts()
print("\n--- Target counts ---")
print(vc.to_string())
default_rate = vc.get(1, 0) / vc.sum()
print(f"Default rate: {default_rate:.3%}")

summary = pd.DataFrame({
    "count": df.count(),
    "missing": df.isnull().sum(),
    "dtype": df.dtypes,
})
summary.to_csv(os.path.join(OUTDIR, "column_summary.csv"))
print("Saved column_summary.csv")

desc = df.describe(include='all', datetime_is_numeric=True).T
desc.to_csv(os.path.join(OUTDIR, "describe.csv"))
print("Saved describe.csv")

corr = df.corr(numeric_only=True)
corr.to_csv(os.path.join(OUTDIR, "correlation_matrix.csv"))
print("Saved correlation_matrix.csv")

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, vmax=1, vmin=-1)
plt.title("Correlation heatmap (numerical)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "corr_heatmap.png"))
plt.close()
print("Saved corr_heatmap.png")

if 'default' in corr.columns:
    corr_with_target = corr['default'].abs().drop('default').sort_values(ascending=False)
    corr_with_target.to_csv(os.path.join(OUTDIR, "corr_with_target.csv"))
    print("\n--- Top features by |corr| with target ---")
    print(corr_with_target.head(12).to_string())
else:
    print("\nNo 'default' in correlation matrix (unexpected).")

top_feats = list(corr_with_target.head(8).index) if 'corr_with_target' in locals() else list(df.columns.drop('default')[:6])
sample = df.sample(n=min(3000, len(df)), random_state=42)

for feat in top_feats:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=sample, x=feat, hue="default", kde=True, element="step", stat="density", common_norm=False)
    plt.title(f"{feat} distribution by default")
    plt.tight_layout()
    fname = os.path.join(OUTDIR, f"hist_{feat}.png")
    plt.savefig(fname)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.boxplot(x="default", y=feat, data=sample)
    plt.title(f"{feat} boxplot by default")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"box_{feat}.png"))
    plt.close()

print(f"Saved histograms & boxplots for top features to {OUTDIR}/")

cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
for c in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=c, data=df, order=df[c].value_counts().index)
    plt.title(f"Counts of {c}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"count_{c}.png"))
    plt.close()
if cat_cols:
    print("Saved categorical count plots:", cat_cols)

print("\nEDA complete. Inspect files in the folder:", OUTDIR)
