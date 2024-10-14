import json
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

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

def tune_model_with_gridsearch(X_train, Y_train, model, param_grid):
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=tscv)
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_, grid_search.best_score_

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2, "MSE": mse}

def store_metrics(train_metrics, test_metrics, val_metrics=None):
    metrics = {
        "Train": train_metrics,
        "Test": test_metrics,
        "Validation": val_metrics if val_metrics else {"RMSE": None, "R2": None, "MSE": None}
    }
    return metrics

def visualize_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.index, y_true, label='Actual', color='blue')
    plt.plot(y_true.index, y_pred, label=f'Predicted ({model_name})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Actual vs. Predicted Prices ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_models(X_train, X_test, Y_train, Y_test):
    metrics_dict = {}

    # Linear Regression tuning
    lr_param_grid = {'fit_intercept': [True, False]}
    lr_model, _ = tune_model_with_gridsearch(X_train, Y_train, LinearRegression(), lr_param_grid)
    
    # Train and test predictions for Linear Regression
    lr_train_pred = lr_model.predict(X_train)
    lr_test_pred = lr_model.predict(X_test)

    # Calculate metrics
    lr_train_metrics = calculate_metrics(Y_train, lr_train_pred)
    lr_test_metrics = calculate_metrics(Y_test, lr_test_pred)

    metrics_dict["Linear Regression"] = store_metrics(lr_train_metrics, lr_test_metrics)

    # Visualize predictions
    visualize_predictions(Y_test, lr_test_pred, "Linear Regression")

    # Random Forest tuning
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_model, _ = tune_model_with_gridsearch(X_train, Y_train, RandomForestRegressor(), rf_param_grid)
    
    # Train and test predictions for Random Forest
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    # Calculate metrics
    rf_train_metrics = calculate_metrics(Y_train, rf_train_pred)
    rf_test_metrics = calculate_metrics(Y_test, rf_test_pred)

    metrics_dict["Random Forest"] = store_metrics(rf_train_metrics, rf_test_metrics)

    # Visualize predictions
    visualize_predictions(Y_test, rf_test_pred, "Random Forest")

    # XGBoost Regressor tuning
    xgb_param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3]
    }
    xgb_model, _ = tune_model_with_gridsearch(X_train, Y_train, XGBRegressor(objective='reg:squarederror'), xgb_param_grid)
    
    # Train and test predictions for XGBoost
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    # Calculate metrics
    xgb_train_metrics = calculate_metrics(Y_train, xgb_train_pred)
    xgb_test_metrics = calculate_metrics(Y_test, xgb_test_pred)

    metrics_dict["XGBoost Regressor"] = store_metrics(xgb_train_metrics, xgb_test_metrics)

    # Visualize predictions
    visualize_predictions(Y_test, xgb_test_pred, "XGBoost Regressor")

    return metrics_dict

if __name__ == "__main__":

    ticker = "PANW"
    start_date = "2015-01-01"
    end_date = "2024-09-10"
    label = 'Target'

    data = yf.download(ticker, start = start_date, end=end_date)
    #plot(data, ticker=ticker)

    data['Target'] = data['Adj Close'].shift(-1) - data['Adj Close']
    data['Returns'] = data['Adj Close'].pct_change()
    data['RSI'] = calculate_RSI(data['Adj Close'], 14)
    data['EMA12'] = data['Adj Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['Price_Change'] = data['Adj Close'].diff()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Diff_Close'] = data['Adj Close'].diff()
    data = data.dropna()

    print(data)

    Y = data['Target']
    X = data.drop(['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close'], axis=1)
    
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    metrics_dict = compare_models(X_train, X_test, Y_train, Y_test)

    # Print metrics in JSON format
    metrics_json = json.dumps(metrics_dict, indent=4)
    print(metrics_json)