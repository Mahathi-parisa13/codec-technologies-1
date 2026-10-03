import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "dataset/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(file_path)

print("Original shape:", df.shape)

# Remove ID columns
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Convert transaction type into numbers
encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

# Separate input and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

print("\nBefore SMOTE:")
print(y.value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Apply SMOTE only to training data
smote = SMOTE(
    random_state=42,
    sampling_strategy=0.2
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

# Save prepared data
X_train_resampled.to_csv(
    "model/X_train.csv",
    index=False
)

y_train_resampled.to_csv(
    "model/y_train.csv",
    index=False
)

X_test.to_csv(
    "model/X_test.csv",
    index=False
)

y_test.to_csv(
    "model/y_test.csv",
    index=False
)

print("\nPreprocessing completed successfully!")