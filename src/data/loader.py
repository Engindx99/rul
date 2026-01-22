import pandas as pd
import os
from src.utils.constants import COLS_Raw

def load_data(data_dir: str, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, test, and RUL data for a specific subset.
    
    Args:
        data_dir: Path to directory containing raw text files.
        subset: Dataset identifier (e.g., 'FD001').
        
    Returns:
        train_df, test_df, rul_df
    """
    train_file = os.path.join(data_dir, f"train_{subset}.txt")
    test_file = os.path.join(data_dir, f"test_{subset}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{subset}.txt")

    # Read CSVs (space separated)
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLS_Raw)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLS_Raw)
    
    # RUL file has only one column (RUL values for test units)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])
    
    return train_df, test_df, rul_df
