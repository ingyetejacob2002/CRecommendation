import pandas as pd
from pathlib import Path

file_path = Path("data/processed/north_central_crop_requirements.csv")

crops = pd.read_csv(file_path)

print("Actual crop names in your file:")
for crop in crops["crop"].dropna().unique():
    print(repr(crop))