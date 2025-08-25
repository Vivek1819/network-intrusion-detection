import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)

    # Drop useless cols
    drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'label']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical cols
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Split X, y
    X = df.drop(columns=['attack_cat'])
    y = LabelEncoder().fit_transform(df['attack_cat'])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
