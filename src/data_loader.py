import pandas as pd

def load_unsw_data(path):
    """
    Loads UNSW-NB15 dataset from a CSV file.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    return pd.read_csv(path)
