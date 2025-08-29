import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Cleans and preprocesses the UNSW-NB15 dataset.

    Steps:
    - Drops unused columns
    - Encodes categorical features
    - Scales numerical features

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        X (np.ndarray): Processed features
        y (np.ndarray): Encoded target labels
        scaler (StandardScaler): fitted scaler (for later use)
    """
    drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical columns
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target
    y = df['attack_cat']
    y = LabelEncoder().fit_transform(y)

    X = df.drop(columns=['attack_cat'])
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def get_binary_target(df):
    return (df['attack_cat'] != 'Normal').astype(int)
