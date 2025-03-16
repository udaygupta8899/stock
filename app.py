import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime, timedelta
from time import sleep
from random import randint
import joblib

# Add the model directory to the path
sys.path.append(os.path.abspath("model"))

# Set page config for wide mode and title (UI remains unchanged)
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide",  # Enable wide mode by default
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    :root {
        --primary: #4FC3F7;
        --background: #0E1117;  /* Dark background */
        --card-bg: rgba(255, 255, 255, 0.05);  /* Transparent card */
        --text-color: #ffffff;
        --hover-color: #4FC3F7;
    }
    .stApp {
        background: var(--background);
        color: var(--text-color);
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .news-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .news-card:hover {
        transform: translateY(-3px);
    }
    .prediction-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .prediction-up {
        color: #4CAF50;
    }
    .prediction-down {
        color: #F44336;
    }
    .prediction-neutral {
        color: #FFC107;
    }
    h1, h2, h3 {
        color: var(--hover-color) !important;
        margin-bottom: 1rem !important;
    }
    a {
        color: var(--hover-color);
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--hover-color) 0%, transparent 100%);
        margin: 2rem 0;
    }
    .st-bb { background-color: transparent; }
    .st-at { background-color: var(--hover-color) !important; }
</style>
""", unsafe_allow_html=True)

# Stock list
all_stocks = {
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Nestle India": "NESTLEIND.NS",
    # Add more stocks if needed...
}

# Map YFinance ticker to dataset symbol
def map_ticker_to_symbol(ticker):
    symbol_map = {
        "INFY.NS": "INFY",
        "HDFCBANK.NS": "HDFCBANK",
        "RELIANCE.NS": "RELIANCE",
        "ICICIBANK.NS": "ICICIBANK",
        "AXISBANK.NS": "AXISBANK",
        "KOTAKBANK.NS": "KOTAKBANK",
        "SBIN.NS": "SBIN",
        "LT.NS": "LT",
        "BAJFINANCE.NS": "BAJFINANCE",
        "HINDUNILVR.NS": "HINDUNILVR",
        "TCS.NS": "TCS",
        "MARUTI.NS": "MARUTI",
        "M&M.NS": "MM",
        "ITC.NS": "ITC",
        "ASIANPAINT.NS": "ASIANPAINT",
        "SUNPHARMA.NS": "SUNPHARMA",
        "DRREDDY.NS": "DRREDDY",
        "TATAMOTORS.NS": "TATAMOTORS",
        "BAJAJFINSV.NS": "BAJAJFINSV",
        "NESTLEIND.NS": "NESTLEIND",
    }
    return symbol_map.get(ticker, None)

# Sidebar Configuration
st.sidebar.title("📈 Stock Dashboard")
st.sidebar.markdown("---")
selected_stock_name = st.sidebar.selectbox(
    "Select Company",
    list(all_stocks.keys()),
    format_func=lambda x: f"{x} ({all_stocks[x]})"
)
selected_stock = all_stocks[selected_stock_name]

st.sidebar.markdown("---")
selected_period = st.sidebar.selectbox(
    "Time Period",
    ["1d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=4
)

st.sidebar.markdown("---")
st.sidebar.caption("Chart Settings")
candlestick_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)
show_predictions = st.sidebar.checkbox("Show ML Predictions", value=True)

# Function to fetch stock data with retries and error handling
@st.cache_data(ttl=600)
def fetch_stock_data(symbol, period):
    retry_count = 3
    for _ in range(retry_count):
        try:
            stock = yf.Ticker(symbol)
            if period == "1h":
                df = stock.history(period="1d", interval="1m")
                if df.empty:
                    st.warning("No data found for the last 1 hour. Trying with broader period.")
                    df = stock.history(period="1d", interval="5m")
            else:
                df = stock.history(period=period)
            info = stock.info
            return df, info
        except Exception as e:
            st.warning(f"Error fetching data (attempting retry): {e}")
            sleep(randint(1, 3))
    st.error("Failed to fetch stock data after multiple attempts.")
    return None, None

# Improved news filtering
def get_relevant_news(stock_name, ticker):
    news_api_key = "813bb17cd2704c12a2acf66732f973bc"  # Replace with your key
    full_name = stock_name
    query = f'"{full_name}" OR "{ticker}"'
    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': news_api_key,
        'from': date_from,
        'qInTitle': stock_name
    }
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        filtered = []
        for article in articles:
            title = article.get('title', '').lower() if article.get('title') else ""
            desc = article.get('description', '').lower() if article.get('description') else ""
            if any([full_name.lower() in title, ticker.lower() in title, full_name.lower() in desc, ticker.lower() in desc]):
                filtered.append(article)
        return filtered[:5]
    except Exception as e:
        st.error(f"News API Error: {e}")
        return []

# Gradient Boosted Model Prediction Function using LightGBM
def get_gb_predictions(stock_symbol, data_df):
    model_path = "model/model.pkl"
    scaler_path = "model/scaler.joblib"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning("Model or scaler files not found.")
        return None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Sort data by date
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    if "Close" not in data_df.columns:
        st.error("Data does not have 'Close' column.")
        return None
    
    # Create features if not already present
    data_df["lag1"] = data_df["Close"].shift(1)
    data_df["lag2"] = data_df["Close"].shift(2)
    data_df["ma5"] = data_df["Close"].rolling(window=5).mean()
    data_df["ma10"] = data_df["Close"].rolling(window=10).mean()
    data_df["day_of_week"] = pd.to_datetime(data_df["Date"]).dt.dayofweek
    data_df = data_df.dropna().reset_index(drop=True)
    if data_df.empty:
        st.error("Not enough data to generate features for prediction.")
        return None
    
    # Use the last row as seed for recursive forecasting
    last_row = data_df.iloc[-1].copy()
    current_date = pd.to_datetime(last_row["Date"])
    current_close = last_row["Close"]
    history = data_df["Close"].values[-10:].tolist()
    
    pred_dates = []
    pred_prices = []
    for i in range(7):
        next_date = current_date + timedelta(days=1)
        # Skip weekends
        while next_date.weekday() > 4:
            next_date += timedelta(days=1)
        lag1 = current_close
        lag2 = history[-2] if len(history) >= 2 else current_close
        ma5 = np.mean(history[-5:]) if len(history) >= 5 else current_close
        ma10 = np.mean(history[-10:]) if len(history) >= 10 else current_close
        day_of_week = next_date.weekday()
        X_new = np.array([[lag1, lag2, ma5, ma10, day_of_week]])
        X_new_scaled = scaler.transform(X_new)
        next_close = model.predict(X_new_scaled)[0]
        pred_dates.append(next_date)
        pred_prices.append(next_close)
        history.append(next_close)
        current_close = next_close
        current_date = next_date
    
    pred_df = pd.DataFrame({
        "Date": pred_dates,
        "Predicted_Close": pred_prices
    })
    pred_df["Predicted_Return"] = pred_df["Predicted_Close"].pct_change().fillna(0)
    return pred_df

# Function to generate sentiment based on predictions
def generate_sentiment_from_predictions(predictions):
    if predictions is None or predictions.empty:
        return None
    first_price = predictions['Predicted_Close'].iloc[0]
    last_price = predictions['Predicted_Close'].iloc[-1]
    overall_change = (last_price - first_price) / first_price * 100
    if overall_change > 2:
        return {"sentiment": "positive", "change": f"+{overall_change:.2f}%", "class": "prediction-up"}
    elif overall_change < -2:
        return {"sentiment": "negative", "change": f"{overall_change:.2f}%", "class": "prediction-down"}
    else:
        return {"sentiment": "neutral", "change": f"{overall_change:.2f}%", "class": "prediction-neutral"}

# Main App
def main():
    st.title(f"{selected_stock_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data Loading
    with st.spinner('Loading market data...'):
        df, info = fetch_stock_data(selected_stock, selected_period)
    if df is None or df.empty:
        st.warning("No data available for the selected stock")
        return

    # Key Metrics Grid
    st.subheader("Key Metrics")
    cols = st.columns(4)
    metrics = [
        ("Current Price", f"₹{df['Close'].iloc[-1]:,.2f}"),
        ("Market Cap", f"₹{info.get('marketCap', 0)/1e7:,.1f} Cr"),
        ("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}"),
        ("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}")
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Price chart with Bollinger Bands
    st.subheader("Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    if candlestick_ma:
        for days, color in [(20, '#FFA726'), (50, '#26C6DA')]:
            ma = df['Close'].rolling(days).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ma,
                name=f'{days} MA',
                line=dict(color=color, width=2)
            ))
    if show_bollinger:
        window = 20
        sma = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma,
            line=dict(color='#FF6F00', width=1.5),
            name='Bollinger Middle (20 SMA)'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=upper_band,
            line=dict(color='#4CAF50', width=1.5),
            name='Upper Band (2σ)',
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=lower_band,
            line=dict(color='#F44336', width=1.5),
            name='Lower Band (2σ)',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))
    fig.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI Chart
    if show_rsi:
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        rsi = calculate_rsi(df)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=rsi,
            line=dict(color='#8A2BE2', width=2),
            name='RSI'
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.update_layout(
            height=400,
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="RSI"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Volume Chart
    st.subheader("Trading Volume")
    fig_vol = go.Figure(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker=dict(color='rgba(255, 99, 132, 0.6)'),
        name="Volume"
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Predictions Section - using GB model
    if show_predictions:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("ML Price Predictions (Next 7 Days)")
        with st.spinner('Generating ML predictions...'):
            predictions = get_gb_predictions(selected_stock, df)
        if predictions is not None and not predictions.empty:
            # Display prediction table
            st.markdown("### Detailed Daily Predictions")
            predictions_table = pd.DataFrame({
                'Date': predictions['Date'].dt.strftime('%Y-%m-%d'),
                'Predicted Close': predictions['Predicted_Close'].map('₹{:,.2f}'.format),
                'Daily Return': predictions['Predicted_Return'].map('{:.2%}'.format)
            })
            st.table(predictions_table)
            
            # Generate sentiment based on predictions
            sentiment = generate_sentiment_from_predictions(predictions)
            if sentiment:
                sentiment_cols = st.columns(3)
                with sentiment_cols[0]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Predicted Trend</h3>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                            {sentiment['sentiment'].title()}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with sentiment_cols[1]:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Expected Change</h3>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                            {sentiment['change']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with sentiment_cols[2]:
                    current_price = df['Close'].iloc[-1]
                    target_price = predictions['Predicted_Close'].iloc[-1]
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>7-Day Target Price</h3>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                            ₹{target_price:,.2f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display prediction line chart
            st.line_chart(predictions.set_index("Date")["Predicted_Close"])
        else:
            st.warning("Predictions are not available for this stock. Please check if the model is trained or if this stock is in the training dataset.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Display News
    st.subheader("Latest News")
    with st.spinner("Loading news..."):
        news_articles = get_relevant_news(selected_stock_name, selected_stock)
    if news_articles:
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            st.markdown(f"""
            <div class="news-card">
                <h3><a href="{url}" target="_blank">{title}</a></h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No news found for the selected stock.")

if __name__ == "__main__":
    main()
