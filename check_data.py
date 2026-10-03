import pandas as pd
import glob

files = glob.glob("dataset/*.csv")

print("CSV files found:")
print(files)

if len(files) == 0:
    print("ERROR: No CSV file found inside dataset folder.")
else:
    file_path = files[0]

    print("\nUsing file:")
    print(file_path)

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFraud count:")
    print(df["isFraud"].value_counts())