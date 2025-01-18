import os
import bs4 as bs
import pickle
import requests
import pandas as pd
import yfinance as yf  
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def save_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia, extracts their symbols,
    and saves them to a pickle file for future use.
    """
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')  
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    
    for row in table.findAll('tr')[1:]:  # Skip the header row
        cols = row.findAll('td')
        if cols:  
            ticker = cols[1].text.strip()  # Extract the ticker symbol
            tickers.append(ticker)
    
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)
    
    print(tickers)
    return tickers


def get_data_from_yahoo(reload_sp500=False):
    """
    Downloads historical stock data for all S&P 500 tickers using Yahoo Finance API.
    The data is stored as CSV files in the 'stock_dfs' directory.
    
    Args:
        reload_sp500 (bool): Whether to reload the S&P 500 tickers list.
    """
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    
    start = dt.datetime(2019, 6, 8)  # Start date for fetching data
    end = dt.datetime.now()  # End date is the current date
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        filepath = f'stock_dfs/{ticker}.csv'
        if not os.path.exists(filepath):  # Skip already downloaded data
            try:
                df = yf.download(ticker, start=start, end=end)
                df.to_csv(filepath)
            except Exception as e:
                print(f"Could not fetch data for {ticker}: {e}")
        else:
            print(f'Already have {ticker}')


def compile_data():
    """
    Combines the 'Close' prices from all stock data into a single DataFrame and saves it.
    The resulting file contains the 'Close' price of each ticker indexed by date.
    """
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv(f'stock_dfs/{ticker}.csv')

            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            else:
                possible_date_column = df.columns[0]  
                print(f"Assuming {possible_date_column} is the Date column.")
                df.set_index(possible_date_column, inplace=True)

            df = df[['Close']]  # Keep only the 'Close' column
            df.columns = [ticker]  # Rename column to the ticker symbol

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')  # Merge data on dates

            if count % 10 == 0:
                print(f"Processed {count} tickers")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("Combining data completed.")
    main_df.to_csv('sp500JoinedCloses.csv')  # Save the combined data


def visualize_data():
    """
    Visualizes the 'Close' price data for a specific ticker (APO) and displays 
    the correlation matrix for all S&P 500 stocks.
    """
    df = pd.read_csv('sp500JoinedCloses.csv', index_col='Price', parse_dates=True)
    
    df = df.apply(pd.to_numeric, errors='coerce')  # Ensure numeric data
    
    if 'APO' in df.columns:
        df['APO'].plot()  # Plot the 'Close' price for 'APO'
    else:
        print("The 'APO' column was not found in the DataFrame.")

    df_corr = df.corr()  # Compute the correlation matrix
    data = df_corr.values
    print(data)



visualize_data()
