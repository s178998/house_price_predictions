from house_price_pipeline.src.model import build_model, train_model, save_model
from house_price_pipeline.src.evaluate import evaluate_model
from house_price_pipeline.src.preprocess import build_preproccessor, transform_data
from house_price_pipeline.src.data_loader import load_data
from house_price_pipeline.src.tuning import tune_model
from house_price_pipeline.src.plots import plot_predictions
import joblib as jb
import pandas as pd
import os

# Load & preprocess
x_train, x_test, y_train, y_test = load_data(r'C:\Users\ayode\Python projects for job\house_price_pipeline\data\raw\california_housing_clean.csv', target_column='MedHouseVal', drop_duplicates=True, test_size=.2, random_state=42)

preprocess = build_preproccessor(x_train)
jb.dump(preprocess, filename="models/preproccessor.pkl")



x_train_processed = transform_data(preprocess, x_train)
x_test_processed = transform_data(preprocess, x_test)



shape = x_train_processed.shape[1]

# Build and train model
best_model = tune_model(x_train_processed, y_train, shape)

y_pred = best_model.predict(x_test_processed).flatten()
avg = y_pred.mean()
print(f"Average medium house price california is: {avg * 100000:.2f}")



mse, r2 = evaluate_model(y_test, y_pred)

best_model.save("models/best_model.keras")

os.makedirs("models", exist_ok=True)
with open ("models/AveragePrice.txt", 'w') as f:
    f.write(str(avg))


plot_predictions(y_test, y_pred)

