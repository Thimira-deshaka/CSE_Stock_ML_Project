import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def run_preprocessing(raw_stock_path, raw_macro_path, output_path):
    """
    Automated pipeline to clean stock and macro data, 
    merge them, and apply scaling for all 4 Macro indicators.
    """
    print("Starting Preprocessing Pipeline (4 Indicators)...")

    # 1. Load Data
    # Stock data: yfinance CSVs usually have two header rows
    cse = pd.read_csv(raw_stock_path, header=[0, 1], index_col=0)
    
    # New Macro data: Already formatted as [Year, Inflation, GDP_Growth, Interest_Rate, Exchange_Rate]
    macro = pd.read_csv(raw_macro_path)

    # Flatten Multi-index columns for CSE data
    cse.columns = cse.columns.get_level_values(0)
    cse.index = pd.to_datetime(cse.index)
    cse = cse.reset_index()

    # 2. Feature Engineering (CSE)
    # Extract Year to merge daily stock data with yearly macro data
    cse['Year'] = cse['Date'].dt.year

    # 3. Merging datasets
    # Daily CSE data meets Yearly Macro data on 'Year'
    df = pd.merge(cse, macro, on='Year', how='left')

    # 4. Normalization (Requirement 1)
    # We now scale all 4 macroeconomic features
    feature_cols = ['Inflation', 'GDP_Growth', 'Interest_Rate', 'Exchange_Rate']
    
    # Drop rows where essential features or the target (Close) are missing
    # This handles any years where World Bank data might be empty
    df = df.dropna(subset=feature_cols + ['Close'])
    
    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- SAVE SCALER ---
    # Crucial for the Streamlit app to transform user inputs correctly
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Success: Scaler saved to models/scaler.pkl")

    # 5. Save Processed Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Filter only necessary columns for the final training CSV
    final_df = df[['Date', 'Close'] + feature_cols]
    final_df.to_csv(output_path, index=False)
    
    print(f"Success! Processed data with 4 features saved to: {output_path}")
    return final_df

if __name__ == "__main__":
    # Define paths relative to the project root
    RAW_STOCK = "data/raw/cse_historical_data.csv"
    RAW_MACRO = "data/raw/srilanka_macro_data.csv"
    PROCESSED_OUT = "data/processed/processed_data.csv"
    
    run_preprocessing(RAW_STOCK, RAW_MACRO, PROCESSED_OUT)