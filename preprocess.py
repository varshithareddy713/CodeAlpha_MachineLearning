import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("uci_credit_default.csv")

df = df.drop_duplicates()
df.to_csv("uci_credit_default_deduped.csv", index=False)

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data preprocessed successfully!")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)