import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.header('Stock Market Predictor')

stock_symbol = st.text_input('Enter Stock Symbol', 'GOOG').upper()
start_date = '2021-01-01'
prediction_years = st.slider('Select Prediction Period (Years)', 1, 4, 2)

today = pd.to_datetime('today').normalize()
end_date = today + pd.DateOffset(years=prediction_years)
end_date_str = end_date.strftime('%Y-%m-%d')

try:
    # Fetch stock data with improved error handling
    stock_data = yf.download(stock_symbol, start_date, end_date_str)

    if stock_data.empty:
        st.error(f"No data found for {stock_symbol}. Please check the symbol and date range.")
    else:
        st.subheader(f'Stock Data for {stock_symbol}')
        st.write(stock_data)

        # Prepare data for Prophet - Ensure 'y' is 1D
        df_prophet = pd.DataFrame({'ds': stock_data.index, 'y': stock_data['Close'].values.ravel()}).reset_index(drop=True)
        df_prophet = df_prophet.dropna(subset=['y'])

        # Train Prophet model
        model = Prophet()
        model.fit(df_prophet)

        # Calculate the future date range using pd.Timedelta
        future_days = (end_date - df_prophet['ds'].max())
        future_dataframe = model.make_future_dataframe(periods=future_days.days) #added .days

        forecast = model.predict(future_dataframe)

        # Rest of your code...
        # ... (Display forecast, plots, etc.) ...
        st.subheader(f'Forecast for {stock_symbol}')
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
        st.write(forecast_display)

        # Display today's forecast if available
        if today in forecast_display.index:
            st.subheader(f"Today's Forecast ({today.strftime('%Y-%m-%d')})")
            st.write(forecast_display.loc[[today]])
        else:
            st.write(f"Forecast for today ({today.strftime('%Y-%m-%d')}) is not in the predicted range.")

        # Plot the forecast
        fig_plotly = go.Figure()
        fig_plotly.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Price'))
        fig_plotly.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price'))
        fig_plotly.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0, 128, 0, 0)'), name='Upper Bound', showlegend=False))
        fig_plotly.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line=dict(color='rgba(0, 128, 0, 0)'), name='Lower Bound', showlegend=False))
        fig_plotly.add_vline(x=today, line_width=2, line_dash="dash", line_color="red", annotation_text="Today")
        fig_plotly.update_layout(title=f'{stock_symbol} Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_plotly)

        # Plot forecast components
        st.subheader('Forecast Components')
        fig_components = model.plot_components(forecast)
        for ax in fig_components.axes:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        st.pyplot(fig_components)

        st.write("### Understanding Forecast Components")
        st.write("""The following charts break down the forecast into its constituent parts, helping you understand the factors influencing the predictions.""")

        with st.expander("Trend: The Long-Term Direction"):
            st.write("""This chart shows the overall long-term trend of the stock price. It represents the general direction the price is moving, independent of seasonal or weekly fluctuations. A positive trend indicates an upward movement over time, while a negative trend suggests a downward movement.""")

        with st.expander("Weekly Seasonality: Patterns Within a Week"):
            st.write("""This chart reveals any recurring patterns that occur within a week. For example, you might see that stock prices tend to be higher on certain days of the week and lower on others. This seasonality captures those weekly cycles.""")

        with st.expander("Yearly Seasonality: Patterns Within a Year"):
            st.write("""This chart shows recurring patterns that happen over the course of a year. For instance, stock prices might be influenced by seasonal factors like holidays, earnings reports, or economic cycles that repeat annually.""")

        if 'holidays' in forecast:
            with st.expander("Holiday Effects: Impact of Special Events"):
                st.write("""This chart (if present) quantifies the impact of specific holidays on the stock price. It shows how the price tends to deviate from the baseline forecast around these dates.""")

except Exception as e:
    st.error(f"An error occurred: {e}")
