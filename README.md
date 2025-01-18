# Python Finance and Machine Learning Project

## Overview
This repository contains two Python scripts, `python_for_finance.py` and `python_for_ml.py`, designed for stock market data analysis and basic machine learning modeling. The scripts fetch and compile stock data, visualize correlations, and train a machine learning model to predict stock movements.

---

## File Descriptions

### `python_for_finance.py`
This script focuses on data collection, processing, and visualization for S&P 500 stocks.

#### Key Functions:
1. **`save_sp500_tickers`**
   - Fetches the list of S&P 500 tickers from Wikipedia and saves them to a file (`sp500tickers.pickle`).
   - Outputs the tickers to the console.

2. **`get_data_from_yahoo(reload_sp500=False)`**
   - Downloads historical stock data from Yahoo Finance for all S&P 500 tickers.
   - Saves individual CSV files for each ticker in the `stock_dfs` directory.

3. **`compile_data`**
   - Compiles `Close` prices for all tickers into a single CSV file (`sp500JoinedCloses.csv`).
   - Uses the date as the index and merges data from all tickers.

4. **`visualize_data`**
   - Loads the compiled data from `sp500JoinedCloses.csv`.
   - Displays the correlation matrix of stock prices and plots a specific ticker (e.g., `APO`) if available.

---

### `python_for_ml.py`
This script applies machine learning techniques to predict stock price movements based on historical data.

#### Key Functions:
1. **`processDataForLabels(ticker)`**
   - Generates percentage changes in stock prices for a specified number of days (`hm_days=7`).
   - Creates new columns representing the changes for the given ticker.

2. **`buy_sell_hold(*args)`**
   - Determines whether to buy (1), sell (-1), or hold (0) based on percentage changes.

3. **`extract_features(ticker)`**
   - Prepares feature matrices (`X`) and target values (`y`) for machine learning.
   - Normalizes data and handles missing or infinite values.

4. **`do_ml(ticker)`**
   - Implements a machine learning pipeline using a Voting Classifier (combining SVM, KNN, and Random Forest).
   - Trains the model and evaluates its accuracy.
   - Outputs predictions and their distribution.

---

## Dependencies
Ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `bs4` (BeautifulSoup)
- `requests`
- `sklearn`
- `pickle`

Install dependencies using pip:
```bash
pip install numpy pandas matplotlib yfinance beautifulsoup4 requests scikit-learn
```

---

## How to Use

### Step 1: Fetch and Compile Data
1. Run `save_sp500_tickers()` to save the list of S&P 500 tickers.
2. Run `get_data_from_yahoo()` to download historical data for all tickers.
3. Run `compile_data()` to compile the data into a single CSV file.

### Step 2: Visualize Data
- Use `visualize_data()` to plot correlations or specific ticker prices.

### Step 3: Train Machine Learning Model
1. Select a ticker to analyze (e.g., `APO`).
2. Call `do_ml(ticker)` to train and test the machine learning model.

---

## Output Files
- **`sp500tickers.pickle`**: Contains the list of S&P 500 tickers.
- **`stock_dfs/{TICKER}.csv`**: Individual CSV files for each ticker's historical data.
- **`sp500JoinedCloses.csv`**: Combined dataset of `Close` prices for all tickers.

---

## Notes and Potential Improvements
- Ensure `sp500tickers.pickle` is present before running `compile_data` or `visualize_data`.
- The correlation visualization in `visualize_data` could be enhanced with heatmaps.
- Consider adding error handling for missing or incomplete data during downloads.
- Experiment with additional features or machine learning models for better accuracy.


---

## Author
devanand98

