import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_RSI(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

def plot(stock_data, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(stock_data.index, stock_data['Close'])
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    ticker = "PANW"
    start_date = "2015-01-01"
    end_date = "2024-09-10"

    data = yf.download(ticker, start = start_date, end=end_date)
    plot(data, ticker=ticker)


    data['RSI'] = calculate_RSI(data['Adj Close'], 14)
    data['EMA12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Price_Change'] = data['Adj Close'].diff()
    data['Volume_Change'] = data['Volume'].pct_change()
    data = data.dropna()

    print(data)

