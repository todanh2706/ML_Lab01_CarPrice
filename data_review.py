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
    
    print("Check null:")
    df.isnull().sum()
    print(df.dtypes)

    print("\nMissing Values:\n", df.isnull().sum())
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

    df.hist(figsize=(15, 10), bins=20)
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

    df['Sector'].unique()

    df['Sector'].value_counts()

    df['Sector'].replace('-1', 'other', inplace=True)
    df['Sector'].value_counts()

    # Top 10 Industries by Number of Job Postings
    top_industries = df['Industry'].value_counts().head(10)

    # Top 10 Industries by Average Salary
    top_salary_industries = df.groupby('Industry')['avg_salary'].mean().sort_values(ascending=False).head(10)

    # Top 10 Sectors by Number of Job Postings
    top_sectors = df['Sector'].value_counts().head(10)

    # Top 10 Sectors by Average Salary
    top_salary_sectors = df.groupby('Sector')['avg_salary'].mean().sort_values(ascending=False).head(10)

    # Visualization 1: Top Industries by Job Postings
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_industries.values, y=top_industries.index, palette="viridis")
    plt.title("Top 10 Industries by Job Postings")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Industry")
    plt.show()

    # Visualization 2: Top Industries by Average Salary
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_salary_industries.values, y=top_salary_industries.index, palette="magma")
    plt.title("Top 10 Industries by Average Salary")
    plt.xlabel("Average Salary (in $K)")
    plt.ylabel("Industry")
    plt.show()

    # Visualization 3: Top Sectors by Job Postings
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_sectors.values, y=top_sectors.index, palette="cubehelix")
    plt.title("Top 10 Sectors by Job Postings")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Sector")
    plt.show()

    # Visualization 4: Top Sectors by Average Salary
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_salary_sectors.values, y=top_salary_sectors.index, palette="coolwarm")
    plt.title("Top 10 Sectors by Average Salary")
    plt.xlabel("Average Salary (in $K)")
    plt.ylabel("Sector")
    plt.show()

    df_copy=df.copy() #independent features
    x = df_copy.drop(columns=['Salary Estimate', 'min_salary', 'max_salary', 'avg_salary'])
    x.head()

    y=df['avg_salary'] #dependent features
    y.head()

    # Select categorical columns for one-hot encoding
    categorical_cols = x.select_dtypes(include=['object']).columns

    # Apply one-hot encoding
    x_encoded = pd.get_dummies(x, columns=categorical_cols, drop_first=True)
    xtrain,xtest,ytrain,ytest=train_test_split(x_encoded,y,test_size=0.2,random_state=42)

    lin_reg=LinearRegression()
    mse=cross_val_score(lin_reg,xtrain,ytrain,scoring='neg_mean_squared_error',cv=5)
    mean_mse=np.mean(mse)
    print(mean_mse)
    lin_reg.fit(xtrain,ytrain)
    ypred=lin_reg.predict(xtest)

    r2_score(ytest,ypred)
    print(r2_score(ytest,ypred))

    print("Done.")

if __name__ == "__main__":
    main()
