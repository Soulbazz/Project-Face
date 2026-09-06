import pandas as pd
import sys

def validate_dataset(file_path, id_column='SEQN'):
    try:
        df = pd.read_csv(file_path)
        print(f"--- Dataset Report: {file_path} ---")
        print(f"Total Rows: {len(df)}")
        
        # 1. Check for Duplicate IDs (Data Leakage risk)
        duplicates = df[df.duplicated(subset=[id_column], keep=False)]
        if not duplicates.empty:
            print(f"[!] WARNING: {len(duplicates)} rows found with duplicate {id_column}s.")
            print(f"    This suggests potential data leakage between Train/Test sets.")
        else:
            print(f"[OK] No duplicate {id_column}s found.")

        # 2. Check for Missing Values in critical columns
        missing = df.isnull().sum()
        if missing.any():
            print(f"[!] WARNING: Missing values detected:\n{missing[missing > 0]}")
        else:
            print("[OK] No missing values found.")

        # 3. Quick Summary
        print(f"--- Integrity Check Complete ---\n")
        
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    # Change the function call at the bottom to use 'name'
    validate_dataset('../data/data.csv', id_column='name')
