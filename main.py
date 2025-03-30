# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)
# Add Date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)

# Flatten MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")

try:
    # Explicit column selection
    cols_to_plot = [col for col in data.columns if col != 'Date']

    # Data validation
    for col in cols_to_plot:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' is not numeric.")

    fig = px.line(data, x='Date', y=cols_to_plot, title='Closing price of the stock', width=1000, height=600)
    st.plotly_chart(fig)
except ValueError as e:
    st.error(f"Plotting error: {e}")
    # Fallback: plot only the first numeric column
    first_numeric_col = next((col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])), None)
    if first_numeric_col:
        fig = px.line(data, x='Date', y=first_numeric_col, title=f'Plotting {first_numeric_col} only', width=1000, height=600)
        st.plotly_chart(fig)
    else:
        st.error("No numeric columns found for plotting.")

# Add a select box to choose the column for forecasting
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# Subsetting the data
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# Model selection
models = ['Prophet']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)

if selected_model == 'Prophet':
    # Prophet Model
    st.header('Facebook Prophet')

    # Prepare the data for Prophet
    prophet_data = data[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})

    # Create and fit
