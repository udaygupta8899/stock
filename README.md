# Stock Dashboard with TFT Predictions

An interactive stock dashboard with Temporal Fusion Transformer (TFT) model integration for stock price predictions based on historical data and sentiment analysis.

## Features

- **Interactive Dashboard**: Real-time stock data visualization with candlestick charts, technical indicators, and volume analysis
- **ML-Powered Predictions**: 7-day price predictions using Temporal Fusion Transformer model
- **Sentiment Analysis**: Integration of synthetic sentiment data to enhance prediction accuracy
- **Technical Indicators**: Support for moving averages, Bollinger Bands, and RSI
- **News Integration**: Relevant news display for each selected stock
- **Modern UI**: Clean and responsive design for better user experience

## Setup and Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stock
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Train the TFT model (optional - pre-trained models available):
   ```
   python model/prepare_model.py
   ```

4. Run the dashboard:
   ```
   streamlit run app.py
   ```

## TFT Model Integration

The dashboard integrates a Temporal Fusion Transformer model for stock price prediction:

- **Data-Driven**: Trained on historical stock data from the Dataset folder
- **Sentiment-Enhanced**: Incorporates synthetic sentiment data based on news patterns
- **Multi-Horizon**: Predicts stock prices for the next 7 days
- **Optimized**: Uses Optuna for hyperparameter optimization
- **Explainable**: Provides trend analysis and confidence metrics

For more details on the model, see the [model README](model/README.md).

## Project Structure

```
stock/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── Dataset/                # Stock price datasets
├── model/                  # TFT model implementation
│   ├── data/               # Processed stock data
│   ├── sentiment/          # Sentiment data generator
│   ├── predictions/        # Pre-generated predictions
│   ├── tft_model.py        # Core TFT model implementation
│   └── prepare_model.py    # Model training pipeline
└── README.md               # Project documentation
```

## Usage

1. Select a stock from the dropdown menu in the sidebar
2. Choose the time period for analysis
3. Toggle chart settings (Moving Averages, Bollinger Bands, RSI)
4. Enable/disable ML predictions with the checkbox
5. View price predictions in the dedicated section below the charts

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- pytorch-forecasting
- pytorch-lightning
- pandas, numpy
- plotly
- yfinance
- optuna

See `requirements.txt` for the complete list of dependencies.

## License

[MIT](LICENSE)
