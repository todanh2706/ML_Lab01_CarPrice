# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os

warnings.filterwarnings("ignore")


def load_data(file_path):
    """
    Data loader
    """
    if not os.path.exists(file_path):
        print(f"[ERR] File not found with path: {file_path}")
        return None
    
    print("Loading data...")
    df = pd.read_csv(file_path)
    return df

def main():
    """
    Data description
    """
     
    DATA_PATH = './archive/eda_data.csv'
    
    df = load_data(DATA_PATH)
    
    if df is None:
        return

    print("\n 5 first line:")
    print(df.head())

    print("\n Data information:")
    df.info()

    print("\n Diagram is being created...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df.columns[0])
    plt.title("Sample diagram")
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
