import pandas as pd
import os
import numpy as np

directory = r"" #path to wherever your CSV files are saved (Same as your code mostly)

# List to hold DataFrames
dfs = []

for filename in os.listdir(directory):
    if filename.endswith((".csv")):  # Check if file is an image
        file_path = os.path.join(directory, filename)

        df = pd.read_csv(file_path)
        dfs.append(df)

result = pd.concat(dfs, ignore_index=True)
result.to_csv("Version_2.csv", index=False)
            
    