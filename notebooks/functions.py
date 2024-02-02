# libraries to import
#basic library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

#Install yfinance and import Data
import yfinance as yf
from yfinance import Ticker


################################### functions to obtain data and save data ###################################

# function to obtain data from yfinance

def get_stock_data(tickers: list, start_date: str, end_date: str, selected_variable: str) -> dict:
    """
    fetches historical stock data for given tickers between specified dates
    input: tickers - list of stock tickers
           start_date - date string in format 'dd mm yyyy'
           end_date - date string in format 'dd mm yyyy'
           selected_variable - name of the variable to select from the stock data
    output: dict - dictionary with ticker symbols as keys and corresponding dataframes as values
    """
    start_date = datetime.strptime(start_date, '%d %m %Y')
    end_date = datetime.strptime(end_date, '%d %m %Y')

    data_dict = {}
    for ticker in tickers:
        ticker_obj = Ticker(ticker)
        history = ticker_obj.history(start=start_date, end=end_date)
        data_dict[ticker] = pd.DataFrame(history)[selected_variable]
    
    return data_dict


def save_series_csv(series_dict: dict, dir_path: str, prefix: str) -> None:
    """
    saves series data to csv files
    input: series_dict - dictionary with ticker symbols as keys and corresponding dataframes as values
           dir_path - directory path where the csv files will be saved
           prefix - prefix for the filename of the csv files
    output: None
    """
    for ticker, series in series_dict.items():
        series.to_csv(f'{dir_path}/{prefix}_{ticker}.csv')

################################### data cleanup functions ###################################

def remove_index_timezone(series_dict: dict) -> dict:
    """
    removes timezone from the index of the series data
    input: series_dict - dictionary with ticker symbols as keys and corresponding dataframes as values
    output: dict - dictionary with timezones removed from the index of the dataframes
    """
    formatted_series = {}
    for ticker, series in series_dict.items():
        series.index = series.index.tz_localize(None)
        formatted_series[ticker] = series
    return formatted_series



def resample_data(data_dict: dict) -> dict:
    """
    resamples the data to daily frequency
    input: data_dict - dictionary with ticker symbols as keys and corresponding dataframes as values
    output: dict - dictionary with resampled dataframes
    """
    resampled_data = {}
    for ticker, series in data_dict.items():
        resampled_series = series.resample('D').asfreq()
        resampled_data[ticker] = resampled_series
    return resampled_data

def ffill_data(data_dict: dict) -> dict:
    """
    fills missing values in the data using forward fill method
    input: data_dict - dictionary with ticker symbols as keys and corresponding dataframes as values
    output: dict - dictionary with forward-filled dataframes
    """
    filled_data = {}
    for ticker, series in data_dict.items():
        filled_series = series.ffill()
        filled_data[ticker] = filled_series
    return filled_data


        

################################### functions visualize plots and correlations ###################################

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sktime.utils.plotting import plot_correlations
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt


def plot_time_series(series_dict: dict) -> None:
    """
    plots time series data for each item in the dictionary
    input: series_dict - dictionary with keys as identifiers and values as pandas series
    output: None
    """
    for item, series in series_dict.items():
        series.index = series.index.to_pydatetime()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series.index, series.values, label=item)
        ax.legend()
        date_form = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        plt.show()


def plot_time_series_corr(series_dict: dict) -> None:
    """
    plots correlation matrix for each item in the dictionary
    input: series_dict - dictionary with keys as identifiers and values as pandas series
    output: None
    """
    for item, series in series_dict.items():
        series.index = series.index.to_pydatetime()
        fig, ax = plot_correlations(series, series_title=item)
        plt.show()



def autocorr_plots(df_dict: dict) -> None:
    """
    generates autocorrelation plots for each dataframe in the dictionary
    input: df_dict - dictionary with keys as identifiers and values as pandas dataframes
    output: None
    """
    for ticker, df in df_dict.items():
        fig, ax = plt.subplots(figsize=(6, 3))
        pd.plotting.autocorrelation_plot(df, ax=ax)
        ax.set_title(f'Autocorrelation Plot for {ticker}')
        plt.show()




def plot_acfs(stock_dict: dict) -> None:
    """
    generates autocorrelation function (ACF) plots for each series in the dictionary
    input: stock_dict - dictionary with keys as identifiers and values as pandas series
    output: None
    """
    for item, series in stock_dict.items():
        print(f"Plotting ACF for {item}")
        plot_acf(series)
        plt.show()




def plot_pacfs(stock_dict: dict) -> None:
    """
    generates partial autocorrelation function (PACF) plots for each series in the dictionary
    input: stock_dict - dictionary with keys as identifiers and values as pandas series
    output: None
    """
    for item, series in stock_dict.items():
        print(f"Plotting PACF for {item}")
        plot_pacf(series)
        plt.show()



################################### statistical tests functions ###################################

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import pandas as pd

def adf_test(dict: dict) -> pd.DataFrame:
    """
    performs augmented dickey fuller test on each series in the dictionary
    input: series_dict - dictionary with keys as identifiers and values as pandas series
    output: pd.DataFrame - dataframe containing the results of the adf tests
    """
    results = []
    for ticker, series in dict.items():
        adfuller_test = adfuller(
            series,
            maxlag=None, 
            regression='c', 
            autolag='AIC', 
            store=False, 
            regresults=False)

        result = {
            'ticker': ticker,
            'ADF Statistic': adfuller_test[0],
            'p-value': adfuller_test[1],
            'used_lag': adfuller_test[2],
            'nobs': adfuller_test[3]
        }

        for key, value in adfuller_test[4].items():
            result[f'Critical Value {key}'] = value

        results.append(result)

    return pd.DataFrame(results)

    from statsmodels.tsa.stattools import kpss

def kpss_test(series_dict: dict) -> pd.DataFrame:
    """
    performs kpss test on each series in the dictionary
    input: series_dict - dictionary with keys as identifiers and values as pandas series
    output: pd.DataFrame - dataframe containing the results of the kpss tests
    """
    results = []
    for ticker, series in series_dict.items():
        kpss_result = kpss(
            series,
            regression='c', 
            nlags='auto', 
            store=True)

        result = {
            'ticker': ticker,
            'KPSS Statistic': kpss_result[0],
            'p-value': kpss_result[1],
            'used_lag': kpss_result[2]
        }

        results.append(result)

    return pd.DataFrame(results)

################################### evaluation functions ###################################



from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def calculate_error_metrics(model_name: str, y_test: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    calculates error metrics (RMSE, MAE, MAPE) for a given model prediction
    input: model_name - string representing the name of the model
           y_test - pandas Series of true target values
           y_pred - pandas Series of predicted target values
    output: pd.DataFrame - dataframe containing the calculated error metrics
    """
    # Calculate the errors
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Create a DataFrame with the results
    df = pd.DataFrame({
        'Model': [model_name],
        'RMSE': [rmse],
        'MAE': [mae],
        'MAPE': [mape]
    })

    return df

################################### modelling functions ###################################


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_and_print_prediction(y_train: pd.Series, y_test: pd.Series, y_pred: pd.Series, train_days_show: int = 365) -> None:
    """
    plots training, testing, and predicted data, prints first prediction
    input: y_train - pandas Series of training target values
           y_test - pandas Series of testing target values
           y_pred - pandas Series of predicted target values
           train_days_show - number of days from the end of the training set to show (default is length of y_train)
    output: None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.iloc[-train_days_show:].index, y_train.iloc[-train_days_show:], label='Train - Series 1')
    plt.plot(y_test.index, y_test, label='Test - Series 1')
    plt.plot(y_pred.index, y_pred, label='Predictions - Series 1')

    # Format x-axis to display dates
    ax = plt.gca()
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)

    # Add gridlines for each year
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(True, which='both')

    plt.legend()
    plt.show()

    print(y_pred.iloc[0])





















