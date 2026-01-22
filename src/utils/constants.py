# Column definitions for CMAPSS Dataset

INDEX_COLS = ["unit", "time"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

COLS_Raw = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors usually discarded in FD001 due to constant values (from literature)
# We will verify this in EDA, but keeping a constant list is good practice.
CONSTANT_SENSORS_FD001 = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]
