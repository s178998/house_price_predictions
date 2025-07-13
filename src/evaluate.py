from sklearn.metrics import  r2_score, mean_squared_error

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"mse: {mse:.2f}")
    print(f"r2 score: {r2:.2f}")

    for actual, pred in zip(y_true[:10], y_pred[:10]):
        print(f"Actual: {actual:.2f}, Predictions: {pred:.2f}")
    return mse, r2