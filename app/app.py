import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

st.title("🏠 House Price Prediction")

# Load preprocessor & model
preprocessor = joblib.load("models/preproccessor.pkl")
model = tf.keras.models.load_model("models/best_model.keras")
try:
    with open("models/AveragePrice.txt", 'r') as f:
        average_price = float(f.read())
except FileNotFoundError:
    average_price = None
        

st.write("### Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")



if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Input Data", df.head(100))



    # Preprocess
    X_processed = preprocessor.transform(df)

    # Predict
    predictions = model.predict(X_processed).flatten()
    df['PredictedPrice'] = predictions

    st.write("### Predictions (1.0 = 100,000)", df[['PredictedPrice']])
    if average_price is not None:
        st.markdown(f"### Average Median house value: {average_price:.4f}(${average_price * 100000:.2f})")

    st.download_button(
        label="Download predictions as CSV",
        data=df.to_csv(index=True).encode(),
        file_name='predictions.csv',
        mime='text/csv'
    )
