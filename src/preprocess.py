import pandas as pd
from src.config import *

def load_raw_data():
    df = pd.read_csv(RAW_DATA)
    return df

def clean_data(df):
    na_counter = df.isna().sum().sum()
    if na_counter == 0:
        print("---No NULLS found---")
        return df
    else:
        print(f"---{na_counter} NULLS found!!---")
        pass

def save_cleaned_data(df):
    df.to_csv(PROCESSED_DATA / "cleaned_data.csv")

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_cleaned = clean_data(df_raw)
    save_cleaned_data(df_cleaned)

    print("---Processed data saved to {PROCESSED_DATA}")