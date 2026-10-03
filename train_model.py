import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier


# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = "dataset/PS_20174392719_1491204439457_log.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =====================================================
# 2. REMOVE ID COLUMNS
# =====================================================

df = df.drop(["nameOrig", "nameDest"], axis=1)


# =====================================================
# 3. CONVERT TRANSACTION TYPE INTO NUMBERS
# =====================================================

encoder = LabelEncoder()

df["type"] = encoder.fit_transform(df["type"])


# =====================================================
# 4. SEPARATE INPUT AND TARGET
# =====================================================

X = df.drop("isFraud", axis=1)

y = df["isFraud"]


print("\nFraud distribution:")
print(y.value_counts())


# =====================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =====================================================
# 6. HANDLE IMBALANCED DATA
# =====================================================

negative = (y_train == 0).sum()

positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nNormal transactions:", negative)
print("Fraud transactions:", positive)

print("Scale positive weight:", scale_pos_weight)


# =====================================================
# 7. CREATE XGBOOST MODEL
# =====================================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)


# =====================================================
# 8. TRAIN MODEL
# =====================================================

print("\nTraining XGBoost model...")
print("Please wait...")

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# =====================================================
# 9. MAKE PREDICTIONS
# =====================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(confusion_matrix(y_test, y_pred))


# =====================================================
# 11. CLASSIFICATION REPORT
# =====================================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(classification_report(y_test, y_pred))


# =====================================================
# 12. SAVE TRAINED MODEL
# =====================================================

print("\nSaving model...")

joblib.dump(model, "fraud_model.pkl")

joblib.dump(encoder, "type_encoder.pkl")


print("\n===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print("Created files:")
print("1. fraud_model.pkl")
print("2. type_encoder.pkl")

print("\nFraud Detection Model is ready!")