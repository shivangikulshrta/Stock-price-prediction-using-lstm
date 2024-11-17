import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMCell
import os
import pickle

# Define function for the stock prediction page
def stock_prediction_page():
    # Set page configuration

    # Load custom CSS for styling
    def load_css():
        css = """
        <style>
            body {
                background-color: #F0F8FF;
                color: #161D6F;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            h1, h2, h3 {
                color: #0B2F9F;
                animation: fadeInDown 1s ease;
            }
            .stButton>button {
                background-color: #0B2F9F;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }
            .stButton>button:hover {
                background-color: #161D6F;
                transform: scale(1.1);
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
            }
            .dataframe {
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
                animation: fadeIn 1s ease;
            }
            th, td {
                border: 1px solid #D3D3D3;
                padding: 10px;
                text-align: center;
            }
            th {
                background-color: #ADD8E6;
                color: #333;
            }
            td {
                background-color: #F5F5F5;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .prediction-graph {
                margin-top: 30px;
                border: 1px solid #0B2F9F;
                border-radius: 8px;
                padding: 20px;
                background-color: white;
                box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
                animation: fadeInUp 1.5s ease;
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .metric-container {
                display: flex;
                justify-content: space-around;
                margin-top: 20px;
                animation: fadeIn 2s ease;
            }
            .metric-box {
                background-color: #0B2F9F;
                color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
                text-align: center;
                font-size: 18px;
                transition: transform 0.3s ease;
            }
            .metric-box:hover {
                transform: scale(1.1);
                background-color: #161D6F;
            }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    load_css()

    # Predefined list of stocks
    predefined_stocks = {
        "Apple_1_stock": "result_dataset.csv",
        "Apple_2_stock": "result_dataset_more_cols_of_one_company.csv",
        "Microsoft_stock": "microsoft_stock_data.csv",
        "Alphabet_stock": "alphabet_stock_data.csv",
        "Amazon_stock": "amazon_stock_data.csv",
        "Apple Inc_stock": "apple_stock_data.csv",
        "Jpm_stock": "jpm_stock_data.csv",
        "Netflix_stock": "netflix_stock_data.csv",
        "Nvidia_stock": "nvidia_stock_data.csv",
        "Meta_stock": "meta_stock_data.csv",
        "Tesla_stock": "tesla_stock_data.csv",
        "Real time": "result_dataset_last_7_days.csv",
    }

    # Main title
    st.title("Stock Price Prediction using LSTM Model")

    # Dropdown for selecting a predefined stock
    selected_stock = st.selectbox("Select a stock", list(predefined_stocks.keys()))

    # Load and initialize the LSTM model
    target_col = "Adj Close"
    model_file = "lstm_trained_model.pkl"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        st.write("Loaded pre-trained model.")

    # Load the selected stock data
    if selected_stock:
        file_path = predefined_stocks[selected_stock]
        df = pd.read_csv(file_path)
        df = df.dropna()

        # Select input columns for prediction
        input_cols = st.multiselect("Select Input Columns", df.columns, default=["Open", "High", "Low", "Close", "Volume"])

        if input_cols:
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            # Scaling data
            input_data = df[input_cols].values
            target_data = df[[target_col]].values

            scale_val_array = df[[target_col]].to_numpy()
            min_scale_val = min(scale_val_array)
            min_scale_val = min_scale_val[0]
            max_scale_val = max(scale_val_array)
            max_scale_val = max_scale_val[0]

            input_scaled = scaler_x.fit_transform(input_data).T
            target_scaled = scaler_y.fit_transform(target_data).T

            # Model prediction
            predictions_scaled = model.predict(input_scaled).T
            predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
            actuals = target_data.flatten()

            # Add predictions to the dataframe
            df["Predicted Prices"] = predictions

            # Filter out NaN values
            mask = ~np.isnan(predictions) & ~np.isnan(actuals)
            actuals, predictions = actuals[mask], predictions[mask]

            # Display the updated table with predicted prices
            st.write(f"### Data Preview for {selected_stock} with Predicted Prices")
            st.write(df.head())  # Display first few rows with the new column added


            max_points = 500000
            if len(actuals) > max_points:
                actuals_to_plot = actuals[-max_points:]
                predictions_to_plot = predictions[-max_points:]
            else:
                actuals_to_plot = actuals
                predictions_to_plot = predictions
                
            # Scale the data to match the specified range for consistent plotting
            plot_scaler = MinMaxScaler(feature_range=(min_scale_val,max_scale_val))
            predictions = plot_scaler.fit_transform(predictions_to_plot.reshape(-1, 1)).flatten()

            # Plotting actual vs. predicted prices without altering scaling
            st.write("### Prediction Results")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(actuals, label="Actual", color="#0B2F9F")
            ax.plot(predictions, label="Predicted", color="orange")
            ax.set_title("Actual vs Predicted Adjusted Closing Prices")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Adjusted Closing Price")
            ax.legend()
            st.pyplot(fig)

            # Model accuracy metrics
            st.write("### Model Accuracy Metrics")
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"**R-squared (R²):** {r2:.4f}")
    else:
        st.info("Select a stock to view its predictions.")


# About Page
def about_page():
    st.title("About")
    st.write("""
    This app provides stock price predictions using an LSTM model.
    You can select a stock from the dropdown menu and view the predicted prices, along with a comparison of actual vs. predicted stock prices.
    """)

# Main Page
def main():
    # Add a navigation bar at the top
    navigation = st.radio("Navigation", ["Home", "Stock Prediction", "About"])

    if navigation == "Home":
        st.title("Welcome to the Stock Prediction App!")
        st.write("This app predicts stock prices using an LSTM model and visualizes the results.")

    elif navigation == "Stock Prediction":
        stock_prediction_page()
        
    elif navigation == "About":
        about_page()

# Run the app
if __name__ == "__main__":
    main()
