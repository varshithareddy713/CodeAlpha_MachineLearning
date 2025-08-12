from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Fetching UCI dataset (ID = 350)...")
ds = fetch_ucirepo(id=350)

X = ds.data.features
y = ds.data.targets

df = X.copy()
df['default'] = y

filename = 'uci_credit_default.csv'
df.to_csv(filename, index=False)
print(f"Dataset saved to {filename} — shape: {df.shape}")
