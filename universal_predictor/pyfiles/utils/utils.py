import pandas as pd
import numpy as np
import random
from datetime import datetime
from scipy.signal import argrelextrema
import plotly.graph_objs as go
import pandas as pd

def plot_decisions_with_markers(decisions, datetimes, closes):
    # Create a DataFrame from the provided lists
    df = pd.DataFrame({
        'DateTime': pd.to_datetime(datetimes),
        'Close': closes,
        'Decision': decisions
    })
    
    # Map decisions to colors, labels, and symbols
    decision_colors = {0: 'green', 1: 'blue', 2: 'red'}
    decision_labels = {0: 'Buy', 1: 'Hold', 2: 'Sell'}
    decision_symbols = {0: 'triangle-up', 1: 'circle', 2: 'x'}
    
    # Create the base plot
    fig = go.Figure()
    
    # Add the Close price line
    fig.add_trace(go.Scatter(x=df['DateTime'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')))
    
    # Add decision markers with different shapes and colors
    for decision in decision_colors.keys():
        decision_df = df[df['Decision'] == decision]
        fig.add_trace(go.Scatter(
            x=decision_df['DateTime'],
            y=decision_df['Close'],
            mode='markers',
            name=decision_labels[decision],
            marker=dict(color=decision_colors[decision], size=7, symbol=decision_symbols[decision])
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title='Decisions on Close Price Over Time',
        xaxis_title='DateTime',
        yaxis_title='Close Price',
        legend_title='Decisions',
        template='plotly_white'
    )
    
    fig.show()

def balance_classes(df, target_column):
    """
    Balances classes in a dataframe by deleting rows to make the target classes near equal in quantity,
    sampling the most recent data.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be balanced.
    target_column (str): The name of the target column in the dataframe.

    Returns:
    pd.DataFrame: A dataframe with balanced classes.
    """
    # Count the number of instances of each class
    class_counts = df[target_column].value_counts()
    
    # Find the minimum class count
    min_class_count = class_counts.min()
    
    # Sample rows to balance the classes from the end of the dataframe
    balanced_df = pd.concat([
        df[df[target_column] == cls].tail(min_class_count)
        for cls in class_counts.index
    ])
    
    # Shuffle the balanced dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df


def add_extrema_targets(df, window_size=2):
    # Identify local minima (bottoms)
    df['min'] = df.iloc[argrelextrema(df['CLOSE'].values, np.less_equal, order=window_size)[0]]['CLOSE']

    # Identify local maxima (peaks)
    df['max'] = df.iloc[argrelextrema(df['CLOSE'].values, np.greater_equal, order=window_size)[0]]['CLOSE']

    # Initialize the TARGET column with 'hold'
    df['TARGET'] = 'hold'

    # Classify as 'buy' at local minima and 'sell' at local maxima
    df.loc[~df['min'].isna(), 'TARGET'] = 'buy'
    df.loc[~df['max'].isna(), 'TARGET'] = 'sell'
    
    # Drop the temporary columns
    df.drop(columns=['min', 'max'], inplace=True)
    
    return df

def create_n_features(df, n):
        """
        Transforms the stock data dataframe to have n hours of features for each column and the target as the n+1-th hour's stock close price.

        :param df: DataFrame with a datetime index
        :param n: Number of hours to use as features
        :return: Transformed DataFrame with features and target
        """
        
        feature_columns = [df.drop(columns=['TARGET'])]
        
        # Create the columns for the features
        for i in range(1, n+1):
            shifted_df = df.drop(columns=['DATETIME', 'TARGET']).shift(i).add_suffix(f'_t-{i}')
            feature_columns.append(shifted_df)
        
        # Combine all shifted dataframes
        feature_df = pd.concat(feature_columns, axis=1)
        
        feature_df['TARGET'] = df['TARGET'].tolist()
    
        feature_df = feature_df.dropna()
        
        return feature_df

def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)

def load_data(file_path):
    """
    Load the data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data by converting date and time columns, renaming columns, removing invalid data, and sorting by datetime.
    
    Parameters:
    df (pd.DataFrame): Raw data.
    
    Returns:
    pd.DataFrame: Preprocessed data.
    """
    
    df.columns = [x.upper() for x in df.columns]
        
    # Ensure all numeric columns are in the correct format
    numeric_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Remove any rows with missing or invalid data
    df = df.dropna()

    # Sort by datetime in ascending order
    df = df.sort_values(by='DATETIME').reset_index(drop=True)

    # Add time-based features
    df['HOUR'] = df['DATETIME'].dt.hour
    df['MINUTE'] = df['DATETIME'].dt.minute
    df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek

    return df

def add_technical_indicators(df, use_volume=False):
    """
    Add technical indicators to the data.
    
    Parameters:
    df (pd.DataFrame): Preprocessed data.
    use_volume (bool): Whether to include volume-based indicators.
    
    Returns:
    pd.DataFrame: Data with technical indicators.
    """
    import ta
    
    # Add moving averages
    df['MA_5'] = ta.trend.sma_indicator(df['CLOSE'], window=5)
    df['MA_10'] = ta.trend.sma_indicator(df['CLOSE'], window=10)
    df['MA_20'] = ta.trend.sma_indicator(df['CLOSE'], window=20)
    df['EMA_5'] = ta.trend.ema_indicator(df['CLOSE'], window=5)
    df['EMA_10'] = ta.trend.ema_indicator(df['CLOSE'], window=10)
    df['EMA_20'] = ta.trend.ema_indicator(df['CLOSE'], window=20)
    
    # Add Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['CLOSE'])
    
    # Add Moving Average Convergence Divergence (MACD)
    df['MACD'] = ta.trend.macd(df['CLOSE'])
    df['MACD_SIGNAL'] = ta.trend.macd_signal(df['CLOSE'])
    df['MACD_DIFF'] = ta.trend.macd_diff(df['CLOSE'])
    
    # Add Bollinger Bands
    bb = ta.volatility.BollingerBands(df['CLOSE'])
    df['BB_HIGH'] = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
    df['BB_MAVG'] = bb.bollinger_mavg()
    df['BB_WIDTH'] = bb.bollinger_wband()
    
    # Add Stochastic Oscillator
    df['STOCH_K'] = ta.momentum.stoch(df['HIGH'], df['LOW'], df['CLOSE'])
    df['STOCH_D'] = ta.momentum.stoch_signal(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # Add Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # Add Commodity Channel Index (CCI)
    df['CCI'] = ta.trend.cci(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # Add True Strength Index (TSI)
    df['TSI'] = ta.momentum.tsi(df['CLOSE'])
    
    # Add Ultimate Oscillator
    df['UO'] = ta.momentum.ultimate_oscillator(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # Add Williams %R
    df['WILLIAMS_R'] = ta.momentum.williams_r(df['HIGH'], df['LOW'], df['CLOSE'])
    
    # Add Keltner Channel
    kc = ta.volatility.KeltnerChannel(df['HIGH'], df['LOW'], df['CLOSE'])
    df['KC_HIGH'] = kc.keltner_channel_hband()
    df['KC_LOW'] = kc.keltner_channel_lband()
    #df['KC_MAVG'] = kc.keltner_channel_mavg()
    
    # Add Donchian Channel
    dc = ta.volatility.DonchianChannel(df['HIGH'], df['LOW'], df['CLOSE'])
    df['DC_HIGH'] = dc.donchian_channel_hband()
    df['DC_LOW'] = dc.donchian_channel_lband()
    df['DC_MAVG'] = dc.donchian_channel_mband()
    
    # Add Percentage Price Oscillator (PPO)
    df['PPO'] = ta.momentum.ppo(df['CLOSE'])
    
    if use_volume:
        # Add On-Balance Volume (OBV)
        df['OBV'] = ta.volume.on_balance_volume(df['CLOSE'], df['VOLUME'])
        
        # Add Accumulation/Distribution Index (ADI)
        df['ADI'] = ta.volume.acc_dist_index(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'])

        # Add Chaikin Money Flow (CMF)
        df['CMF'] = ta.volume.chaikin_money_flow(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'])
        
        # Add Force Index
        df['FORCE_INDEX'] = ta.volume.force_index(df['CLOSE'], df['VOLUME'])
        
        # Add Money Flow Index (MFI)
        df['MFI'] = ta.volume.money_flow_index(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'])
        
        # Add Ease of Movement (EOM)
        df['EOM'] = ta.volume.ease_of_movement(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'])
        
        # Add Volume Price Trend (VPT)
        df['VPT'] = ta.volume.volume_price_trend(df['CLOSE'], df['VOLUME'])
        
        # Add Negative Volume Index (NVI)
        df['NVI'] = ta.volume.negative_volume_index(df['CLOSE'], df['VOLUME'])
    else:
        df = df.drop(columns=['VOLUME'])
    
    # Add categorical indicators
    df['BULLISH'] = df['CLOSE'] > df['MA_20']
    df['BEARISH'] = df['CLOSE'] < df['MA_20']
    df['OVERBOUGHT_RSI'] = df['RSI'] > 70
    df['OVERSOLD_RSI'] = df['RSI'] < 30
    df['MACD_BULLISH'] = df['MACD'] > df['MACD_SIGNAL']
    df['MACD_BEARISH'] = df['MACD'] < df['MACD_SIGNAL']
    df['BB_BANDWIDTH_HIGH'] = df['BB_WIDTH'] > df['BB_WIDTH'].rolling(window=20).mean()
    df['BB_BANDWIDTH_LOW'] = df['BB_WIDTH'] < df['BB_WIDTH'].rolling(window=20).mean()
    df['STOCH_OVERBOUGHT'] = df['STOCH_K'] > 80
    df['STOCH_OVERSOLD'] = df['STOCH_K'] < 20
    df['TSI_BULLISH'] = df['TSI'] > 0
    df['TSI_BEARISH'] = df['TSI'] < 0
    df['UO_OVERBOUGHT'] = df['UO'] > 70
    df['UO_OVERSOLD'] = df['UO'] < 30
    
    return df


def add_holidays(df):
    """
    Add a column indicating if a date is a Russian holiday or important day.
    
    Parameters:
    df (pd.DataFrame): Data with a DATETIME column.
    
    Returns:
    pd.DataFrame: Data with an additional column for holidays.
    """
    import holidays
    # Define the Russian holidays
    ru_holidays = holidays.Russia()

    # Add a column for holidays
    df['HOLIDAY'] = df['DATETIME'].dt.date.apply(lambda x: 1 if x in ru_holidays else 0)

    return df

def drop_nans(df):
    """
    Drop rows with any NaN values.
    
    Parameters:
    df (pd.DataFrame): Data with possible NaN values.
    
    Returns:
    pd.DataFrame: Data without NaN values.
    """
    return df.dropna()

import plotly.graph_objects as go

def plot_stock_close(data, title="Stock Close Prices Over Time"):
    """
    Plot stock close prices over time using Plotly.
    
    Parameters:
    data (pd.DataFrame): Data containing 'DATETIME' and 'CLOSE' columns.
    title (str): Title of the plot.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['DATETIME'],
        y=data['CLOSE'],
        mode='lines',
        name='Close Price'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark'
    )

    fig.show()
    
import plotly.graph_objs as go
import plotly.express as px

def plot_predictions(targets, predictions, title='Stock Price Predictions'):
    """
    Plots the actual targets and predictions using Plotly.

    :param targets: Series or list of actual target values
    :param predictions: Series or list of predicted values
    :param title: Title of the plot
    """
    # Create a dataframe from the targets and predictions
    data = pd.DataFrame({'Actual': targets, 'Predicted': predictions})
    
    # Create the plot
    fig = px.line(data, 
                  title=title, 
                  labels={'value': 'Percentage Increase', 'index': 'Time'},
                  template='plotly_white')

    fig.add_trace(go.Scatter(x=data.index, y=data['Actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Predicted'], mode='lines', name='Predicted'))

    # Customize the layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Percentage Increase',
        legend_title='Legend',
        font=dict(
            family='Arial, sans-serif',
            size=14,
            color='black'
        )
    )
    
    fig.show()
    
import os
import zipfile

def extract_zip_files(src_dir, dest_dir):
    """
    Extracts all .zip files from src_dir to dest_dir, ignoring .gz files.

    :param src_dir: Source directory containing .zip files
    :param dest_dir: Destination directory where the files will be extracted
    """
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Iterate over all files in the source directory
    for file_name in os.listdir(src_dir):
        # Check if the file is a .zip file and not a .gz file
        if file_name.endswith('.zip') and not file_name.endswith('.gz'):
            src_file_path = os.path.join(src_dir, file_name)
            
            # Open the .zip file and extract it
            with zipfile.ZipFile(src_file_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            
            print(f'Extracted {file_name} to {dest_dir}')
