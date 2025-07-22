#data training script for a machine learning model
import pandas as pd

def data_cleaning():
    chunksize=10000
    df = pd.read_csv('data/data.csv', chunksize=chunksize)
    data_df=None
    for data_df in df:
        data_df.dropna(inplace=True)
        data_df.drop_duplicates(keep='first', inplace=True)
        data_df.fillna(data_df.median(numeric_only=True), inplace=True) 
        data_df.fillna("Unknown", inplace=True)
        data_df.drop(columns=data_df.select_dtypes(include=['object', 'category']).columns, inplace=True) 
        data_df.to_csv("data/data_cleaned.csv", index=False)
    print("Data cleaning completed and cleaned datasets saved.")
    return data_df
