import pandas as pd
import lightgbm as lgb
import pickle
import os

def train():
    # Load the cleaned and merged data
    processed_path = 'data/processed/processed_data.csv'
    if not os.path.exists(processed_path):
        print("Error: Processed data not found. Please run preprocess.py first.")
        return

    df = pd.read_csv(processed_path)
    
    # 4-Indicator Feature Set
    X = df[['Inflation', 'GDP_Growth', 'Interest_Rate', 'Exchange_Rate']]
    y = df['Close']
    
    print(f"Training LightGBM model with {len(X)} samples and 4 macro features...")

    # LightGBM configuration
    model = lgb.LGBMRegressor(
        n_estimators=1200, 
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        importance_type='gain'
    )
    
    model.fit(X, y)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/cse_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Success: 4-Indicator model saved to {model_path}")

if __name__ == "__main__":
    train()