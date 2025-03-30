import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2021-01-01'

period = st.slider('Select Prediction Period (Years)', 1, 4, 2)
end = pd.to_datetime('today') + pd.DateOffset(years=period)
end = end.strftime('%Y-%m-%d')

try:
    data = yf.download(stock, start, end)

    st.subheader('Stock Data')
    st.write(data)

    df_train = pd.DataFrame({'ds': data.index, 'y': data.Close.values.ravel()})
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train = df_train.dropna(subset=['y'])
    df_train['ds'] = pd.to_datetime(df_train['ds'])

    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=(pd.to_datetime(end) - pd.to_datetime(start)).days)
    forecast = m.predict(future)

    forecast['ds'] = pd.to_datetime(forecast['ds'])
    df_combined = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_train[['ds','y']], on='ds', how='left')
    df_combined = df_combined.rename(columns={'y': 'Original Price', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
    df_combined = df_combined.set_index('ds')

    st.subheader('Forecast Data')
    st.write(df_combined)

    current_date = pd.to_datetime('today').normalize() #Corrected line
    if current_date in df_combined.index:
        current_data = df_combined.loc[current_date]
        current_data_df = pd.DataFrame(current_data).T
        st.subheader(f"Today's Data ({current_date.strftime('%Y-%m-%d')})")
        st.write(current_data_df)
    else:
        st.write(f"No data available for today's date ({current_date.strftime('%Y-%m-%d')}).")

    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Original Price (Training)', hovertemplate='Date: %{x}<br>Price: %{y:.2f}'))
    fig_plotly.add_trace(go.Scatter(x=future['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price', hovertemplate='Date: %{x}<br>Predicted Price: %{y:.2f}'))
    fig_plotly.add_trace(go.Scatter(x=future['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(0, 128, 0, 0.2)'), name='Upper Bound', hoverinfo='none'))
    fig_plotly.add_trace(go.Scatter(x=future['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='rgba(0, 128, 0, 0.2)'), fill='tonexty', fillcolor='rgba(0, 128, 0, 0.2)', name='Lower Bound', hoverinfo='none'))
    fig_plotly.update_layout(title='Original vs. Predicted Stock Price', xaxis_title='Time', yaxis_title='Price', hovermode='x unified')
    st.plotly_chart(fig_plotly)

    fig_components = m.plot_components(forecast)

    # Beautify the components plots
    for ax in fig_components.axes:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    st.subheader('Forecast Components')
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
