import streamlit as st
import base64
from stock_prediction_frontend import stock_prediction_page

# Set up page configuration
st.set_page_config(page_title="📊 Stock Prediction Dashboard", layout="wide")

# Function to load image and convert to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert your image to base64
base64_background = get_base64_of_bin_file("StockMarket.png")

# Define the sidebar for navigation
st.sidebar.title("Navigation")
pages = {
    "Home": "home",
    "Stock Price Prediction": "stock_prediction",
}
selected_page = st.sidebar.radio("Go to", list(pages.keys()))

# Define home page content with a full-page background image
def home_page():
    # CSS for the background image and text styling
    st.markdown(
        f"""
        <style>
            /* Background Image with Scaling Transition */
            .stApp {{
                background-image: url("data:image/png;base64,{base64_background}");
                background-size: 120%; /* Start with slightly zoomed in */
                background-position: center;
                background-repeat: no-repeat;
                animation: scaleBackground 10s infinite alternate ease-in-out; /* Scaling effect */
            }}

            /* Scaling Keyframes */
            @keyframes scaleBackground {{
                0% {{ background-size: 120%; }}
                100% {{ background-size: 130%; }}
            }}

            /* Sidebar Text Styling */
            .css-1d391kg p {{
                color: #0B2F9F; /* Blue color for navigation text */
                font-weight: bold;
                font-size: 18px;
            }}

            /* Fade-in Effect for Content */
            .content-container {{
                background-color: rgba(0, 0, 0, 0.6);
                padding: 50px;
                border-radius: 10px;
                color: white;
                text-align: center;
                max-width: 800px;
                margin: auto;
                margin-top: 20vh;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
                animation: fadeInContent 2s ease-in-out;
            }}

            /* Text Styling */
            h1 {{
                color: #FFD700;
                font-size: 3rem;
                margin-bottom: 10px;
            }}
            p {{
                font-size: 1.2rem;
                line-height: 1.5;
            }}

            /* Content Fade-in Keyframes */
            @keyframes fadeInContent {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
        
        <div class="content-container">
            <h1>Welcome to the Stock Prediction Dashboard</h1>
            <p>Explore our powerful LSTM model to predict stock prices for various companies. 
            Navigate to the Stock Price Prediction page to get started.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Page navigation logic
if selected_page == "Home":
    home_page()
elif selected_page == "Stock Price Prediction":
    stock_prediction_page()
