import pandas as pd
from sklearn.datasets import fetch_california_housing
import os
data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.head(20))

df['ocean_proximity'] = pd.cut(df['AveOccup'], bins=3, labels=['near', 'medium', 'far'])

os.makedirs('data/raw', exist_ok=True)

df.to_csv('data/raw/california_housing_clean.csv', index=False)
print("Cleaned dataset saved to data/raw/california_housing_clean.csv")