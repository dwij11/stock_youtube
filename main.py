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
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
numeric_cols = data.columns.drop('Date')  # Exclude 'Date' column
fig = px.line(data, x='Date', y=numeric_cols, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

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

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Forecast the future values
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.write("Model selected:", selected_model)

# urls of the images
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
medium_url = "https://img.icons8.com/?size=48&id=BzFWSIqh6bCr&format=png"

# redirect urls
github_redirect_url = "https://github.com/Muhammad-Ali-Butt"
twitter_redirect_url = "https://twitter.com/Data_Maestro"
medium_redirect_url = "https://medium.com/@Data_Maestro"

# adding a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0; 
    width: 100%;
    background-color: #f5f5f5;
    color: #000000;
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">Made with ❤️ by Muhammad Ali Butt<a href="{github_redirect_url}"><img src="{github_url}" width="30" height="30"></a>'
            f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="30" height="30"></a>'
            f'<a href="{medium_redirect_url}"><img src="{medium_url}" width="30" height="30"></a> | Credits: Dr.Ammaar Tufail</div>', unsafe_allow_html=True)
