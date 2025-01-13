import os
import bs4 as bs
import pickle
import requests
import pandas as pd
import yfinance as yf  # Use yfinance instead of pandas_datareader
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'html.parser')  # Explicitly specify the parser
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    
    for row in table.findAll('tr')[1:]:  # Skip the header row
        cols = row.findAll('td')
        if cols:  # Ensure there are table data cells
            ticker = cols[1].text.strip()  # Remove extra whitespace
            tickers.append(ticker)
    
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)
    
    print(tickers)
    return tickers

# save_sp500_tickers()


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    
    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        filepath = f'stock_dfs/{ticker}.csv'
        if not os.path.exists(filepath):
            try:
                # Use yfinance to download stock data
                df = yf.download(ticker, start=start, end=end)
                df.to_csv(filepath)
            except Exception as e:
                print(f"Could not fetch data for {ticker}: {e}")
        else:
            print(f'Already have {ticker}')


# save_sp500_tickers()
# get_data_from_yahoo()



import pandas as pd
import pickle

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            # Read the CSV file to inspect its structure
            df = pd.read_csv(f'stock_dfs/{ticker}.csv')

            # Look for potential column that might be the Date
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            else:
                # Try to identify a possible date column by checking the first few columns
                # If the first column is a date, we can assume it's the date column
                possible_date_column = df.columns[0]  # First column might be Date
                print(f"Assuming {possible_date_column} is the Date column.")
                df.set_index(possible_date_column, inplace=True)

            # Ensure only 'Close' column is kept
            df = df[['Close']]  # Keep only the 'Close' column

            # Rename the column to match the ticker
            df.columns = [ticker]

            # Join the dataframe with the main_df using an outer join (keeping all dates)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            # Progress output every 10 tickers
            if count % 10 == 0:
                print(f"Processed {count} tickers")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Output combined dataframe to CSV
    print("Combining data completed.")
    main_df.to_csv('sp500JoinedCloses.csv')

# Call the function to compile data
# compile_data()


def visualize_data():
    df = pd.read_csv('sp500JoinedCloses.csv', index_col='Price',parse_dates=True)  # Ensure 'Date' is the index
    
    df= df.apply(pd.to_numeric, errors= 'coerce')  # Print column names
    if 'APO' in df.columns:
        df['APO'].plot()
        # plt.title("APO Stock Price")
        # plt.show()
    else:
        print("The 'APO' column was not found in the DataFrame.")



    df_corr = df.corr()
    data = df_corr.values
    print(data)
visualize_data()