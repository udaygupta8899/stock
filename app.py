import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
from time import sleep

# Set page config for wide mode and title
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


# Custom CSS for modern UI/UX
st.markdown("""
<style>
    :root {
        --primary: #4FC3F7;
        --background: #0E1117;
        --card-bg: rgba(255, 255, 255, 0.05);
    }

    .stApp {
        background: var(--background);
        color: #ffffff;
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

    .st-bb { background-color: transparent; }
    .st-at { background-color: var(--primary) !important; }

    h1, h2, h3 {
        color: var(--primary) !important;
        margin-bottom: 1rem !important;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--primary) 0%, transparent 100%);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Stock list
all_stocks = {
    "TCS": "TCS.NS",
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
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Transmission": "ADANITRANS.NS",
    "GAIL": "GAIL.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Cipla": "CIPLA.NS",
    "Grasim Industries": "GRASIM.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Titan Company": "TITAN.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    # Add more stocks if needed...
}

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
    index=1
)

st.sidebar.markdown("---")
st.sidebar.caption("Chart Settings")
candlestick_ma = st.sidebar.checkbox("Show Moving Averages", value=True)

# Function to fetch stock data
@st.cache_data(ttl=600)
def fetch_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

# Improved news filtering
def get_relevant_news(stock_name, ticker):
    news_api_key = "813bb17cd2704c12a2acf66732f973bc"  # Replace with your key
    query = f'"{stock_name}" OR "{ticker}"'
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
        
        # Strict relevance filtering
        filtered = []
        for article in articles:
            title = article.get('title', '').lower() if article.get('title') else ""
            desc = article.get('description', '').lower() if article.get('description') else ""
            if any([
                stock_name.lower() in title,
                ticker.lower() in title,
                stock_name.lower() in desc,
                ticker.lower() in desc
            ]):
                filtered.append(article)
        
        return filtered[:5]

    except Exception as e:
        st.error(f"News API Error: {e}")
        return []

# Main App
def main():
    st.title(f"{selected_stock_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data Loading
    with st.spinner('Loading market data...'):
        df, info = fetch_stock_data(selected_stock, selected_period)
        sleep(0.5)

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

    # Interactive Price Chart
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

    fig.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume Chart
    st.subheader("Trading Volume")
    fig_vol = go.Figure(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color='#4FC3F7'
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Company News Section
    st.subheader("Latest News")
    news_articles = get_relevant_news(selected_stock_name, selected_stock)
    
    if not news_articles:
        st.info("No recent news found for this company")
    else:
        for article in news_articles:
            title = article.get('title', '')
            url = article.get('url', '#')
            source = article.get('source', {}).get('name', 'Unknown')
            date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%b %d, %Y')
            
            st.markdown(f"""
            <div class="news-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <small>{source}</small>
                    <small>{date}</small>
                </div>
                <a href="{url}" target="_blank" style="color: #4FC3F7; text-decoration: none;">
                    <h4>{title}</h4>
                </a>
            </div>
            """, unsafe_allow_html=True)
            sleep(0.2)

if __name__ == "__main__":
    main()
