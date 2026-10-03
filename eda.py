import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "dataset/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

# -----------------------------
# 1. Basic Information
# -----------------------------

print("Dataset Shape:")
print(df.shape)

print("\nTransaction Types:")
print(df["type"].value_counts())

# -----------------------------
# 2. Normal vs Fraud
# -----------------------------

print("\nNormal vs Fraud:")
print(df["isFraud"].value_counts())

# -----------------------------
# 3. Fraud Percentage
# -----------------------------

fraud_percentage = df["isFraud"].mean() * 100

print("\nFraud Percentage:")
print(f"{fraud_percentage:.4f}%")

# -----------------------------
# 4. Transaction Type vs Fraud
# -----------------------------

print("\nFraud by Transaction Type:")
print(
    pd.crosstab(
        df["type"],
        df["isFraud"]
    )
)

# -----------------------------
# 5. Plot Normal vs Fraud
# -----------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    x="isFraud",
    data=df
)

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.show()

# -----------------------------
# 6. Fraud by Transaction Type
# -----------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    x="type",
    hue="isFraud",
    data=df
)

plt.title("Fraud by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.show()

# -----------------------------
# 7. Amount Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="isFraud",
    y="amount",
    data=df
)

plt.title("Transaction Amount vs Fraud")
plt.xlabel("Transaction Class")
plt.ylabel("Amount")

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.show()