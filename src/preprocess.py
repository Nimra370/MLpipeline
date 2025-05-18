import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    return pd.read_csv('data/iris.csv')

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
