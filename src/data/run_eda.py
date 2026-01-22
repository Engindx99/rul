import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.getcwd())

from src.data.loader import load_data
from src.utils.constants import SENSOR_COLS, SETTING_COLS
from src.utils.logger import get_logger

logger = get_logger("EDA")

def run_eda(data_path="data/raw", subset="FD001", output_dir="output/plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {data_path} for subset {subset}...")
    try:
        train_df, test_df, rul_df = load_data(data_path, subset)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Train Shape: {train_df.shape}")
    logger.info(f"Test Shape: {test_df.shape}")
    logger.info(f"RUL Shape: {rul_df.shape}")

    # 1. Null check
    nulls = train_df.isnull().sum().sum()
    logger.info(f"Total Nulls in Train: {nulls}")

    # 2. Statistics
    stats = train_df[SENSOR_COLS].describe().transpose()
    logger.info("\nChecking for constant sensors (std=0):")
    constant_sensors = stats[stats['std'] < 1e-6].index.tolist() # Near zero std
    logger.info(f"Constant Sensors: {constant_sensors}")

    # 3. Visualization: Sample Unit Life
    unit_id = 1
    unit_data = train_df[train_df['unit'] == unit_id]
    
    plt.figure(figsize=(15, 10))
    for i, sensor in enumerate(SENSOR_COLS[:9]): # Plot first 9 sensors
        plt.subplot(3, 3, i+1)
        if sensor in constant_sensors:
            color = 'red'
        else:
            color = 'blue'
        plt.plot(unit_data['time'], unit_data[sensor], color=color)
        plt.title(sensor)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"eda_unit_{unit_id}_sensors.png"))
    logger.info(f"Saved sensor plot to {output_dir}")

if __name__ == "__main__":
    run_eda(data_path="c:/Users/engin/deneme/data/raw") # Using absolute path for safety in this env or use relative if CWD set correctly
