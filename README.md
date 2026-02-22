# Sri Lanka CSE Market Intelligence: Macro-Financial Predictor

An Explainable AI (XAI) dashboard designed to predict the **Colombo Stock Exchange (CSE) All Share Price Index (ASPI)** based on fundamental macroeconomic indicators of Sri Lanka.

## 📌 Project Overview

Predicting stock market trends in a volatile economy like Sri Lanka requires more than just historical price analysis. This project uses **Machine Learning (LightGBM)** to find correlations between the ASPI and four critical "fundamental" drivers:

- **Inflation (NCPI/CCPI)**
- **GDP Growth Rate**
- **Interest Rates (SDFR)**
- **Exchange Rate (LKR/USD)**

## 🚀 Key Features

- **Real-time Simulation:** Adjust economic parameters via a sidebar to see immediate market forecasts.
- **Interactive Visuals:** Gauge charts and delta metrics for quick market health assessment.
- **XAI (Explainable AI):** Uses SHAP-inspired logic to tell the user _why_ the market is moving (e.g., highlighting high interest rates as a primary bearish factor).
- **Robust Pipeline:** Automated preprocessing, scaling, and model training.

---

## 🛠️ System Architecture

1. **Data Ingestion:** Historical macro data (2010–2024) merged with CSE daily closing prices.
2. **Preprocessing:** Feature engineering and normalization using `StandardScaler`.
3. **Modeling:** Gradient Boosting via `LightGBM` (chosen for its efficiency with small, tabular datasets).
4. **Deployment:** Interactive UI built with `Streamlit`.

---

## 📊 Model Performance

The current model achieves the following metrics on the test set:

- **R² Score:** `0.54` (Explains 54% of market variance via macro data alone).
- **MAE:** `340.17` index points.
- **RMSE:** `504.95`.

---

## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CSE_Stock_ML_Project.git
cd CSE_Stock_ML_Project

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Download csv file through running 01_data_collection.ipynb file

To ensure the csv data is available:

Go to above file and run.

### 4. Run the Pipeline

To ensure the model is trained on the latest 4-indicator configuration:

```bash
python src/preprocess.py
python src/train_model.py

```

### 5. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py

```

---

## 📂 Project Structure

```text
├── app/
│   └── streamlit_app.py                # Main dashboard code
├── data/
│   ├── raw/                            # Original CSV files
│   └── processed/                      # Merged & scaled data
├── models/
│   ├── cse_model.pkl                   # Trained LightGBM model
│   └── scaler.pkl                      # Saved StandardScaler
├── src/
│   ├── preprocess.py                   # Data cleaning script
│   └── train_model.py                  # ML training script
├── notbooks/                        #Jupyter notebooks for analysis and training.
│   ├── 01_data_collection.ipynb        # Download Csv files
│   ├── 02_data_cleaning.ipynb          # Data preprocessing
│   ├── 03_eda_visuals.ipynb            # data visualizing
│   └── 04_model_training.ipynb         # Model traing through jupyter notebook (Traning and validation)
└── requirements.txt         # Python dependencies

```

## ⚖️ Disclaimer

This is an **academic project**. Stock market predictions are inherently risky. This tool is intended for educational purposes and should not be used as financial advice.

---

**Would you like me to generate a `requirements.txt` file based on your code to make sure the "Installation" section works perfectly?**

# CSE Stock Machine Learning Project

## Overview

This project predicts CSE stock prices using machine learning.

## Structure

- data/: Contains raw and processed data.
- notebooks/: Jupyter notebooks for analysis and training.
- models/: Saved models.
- src/: Source code for preprocessing and training pipelines.
- app/: Streamlit application.
