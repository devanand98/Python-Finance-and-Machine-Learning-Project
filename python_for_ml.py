from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def processDataForLabels(ticker):
    """
    Prepares the data by calculating percentage changes for the specified ticker over
    the next 7 days (default). Adds new columns to the DataFrame for each of these days.
    
    Args:
        ticker (str): The stock ticker to process.

    Returns:
        list: A list of all tickers in the dataset.
        DataFrame: The processed DataFrame with percentage change columns.
    """
    hm_days = 7  # Number of future days to calculate percentage changes
    df = pd.read_csv('sp500JoinedCloses.csv', index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors='coerce')  # Ensure numeric data
    tickers = df.columns.values.tolist()  # Get list of all tickers
    df.fillna(0, inplace=True)  # Replace NaN values with 0

    # Calculate percentage changes for the specified number of days
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)  # Replace any new NaN values with 0 after calculations
    return tickers, df


def buy_sell_hold(*args):
    """
    Determines a trading signal based on price changes.
    
    Args:
        *args: Price change percentages for the given number of days.

    Returns:
        int: 1 for 'Buy', -1 for 'Sell', 0 for 'Hold'.
    """
    cols = [c for c in args]
    requirement = 0.04  # Threshold for buy/sell decisions
    for col in cols:
        if col > requirement:
            return 1  # Signal to buy
        elif col < -requirement:
            return -1  # Signal to sell
    return 0  # Signal to hold


def extract_features(ticker):
    """
    Extracts features and target labels for machine learning from the dataset.
    
    Args:
        ticker (str): The stock ticker to process.

    Returns:
        tuple: Feature matrix (X), target vector (y), and the modified DataFrame.
    """
    tickers, df = processDataForLabels(ticker)

    # Create a target column by applying the buy_sell_hold function
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    # Print data spread for the target variable
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread: ', Counter(str_vals))
    
    # Handle missing or infinite values
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Create the feature set using percentage change for all tickers
    df_vals = df[[ticker for ticker in tickers]].pct_change()  
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # Feature matrix and target vector
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    """
    Performs machine learning classification using a Voting Classifier to predict
    buy/sell/hold signals for the specified ticker.
    
    Args:
        ticker (str): The stock ticker to process.

    Returns:
        float: The accuracy of the model on the test dataset.
    """
    X, y, df = extract_features(ticker)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
    
    # Define a VotingClassifier with multiple models
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    # Train the classifier
    clf.fit(X_train, y_train)

    # Calculate the accuracy of the model
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)

    # Print the predicted spread of buy/sell/hold signals
    predictions = clf.predict(X_test)
    print('Predicted spread: ', Counter(map(int, predictions)))

    return confidence


# Example usage
do_ml('APO')




 