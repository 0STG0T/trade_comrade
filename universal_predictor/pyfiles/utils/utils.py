import pandas as pd
import numpy as np
import random
from datetime import datetime
from scipy.signal import argrelextrema
import plotly.graph_objs as go
import pandas as pd
import talib
import pywt

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
    numeric_columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
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

import talib
import numpy as np

def add_moving_averages(df):
    """
    Add moving averages to the DataFrame.
    """
    df['MA_3'] = talib.SMA(df['CLOSE'], timeperiod=3)
    df['MA_5'] = talib.SMA(df['CLOSE'], timeperiod=5)
    df['EMA_3'] = talib.EMA(df['CLOSE'], timeperiod=3)
    df['EMA_5'] = talib.EMA(df['CLOSE'], timeperiod=5)
    return df

def add_momentum_indicators(df):
    """
    Add momentum indicators to the DataFrame.
    """
    df['RSI'] = talib.RSI(df['CLOSE'], timeperiod=7)
    df['MACD'], df['MACD_SIGNAL'], df['MACD_DIFF'] = talib.MACD(df['CLOSE'], fastperiod=6, slowperiod=13, signalperiod=5)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['HIGH'], df['LOW'], df['CLOSE'], 
                                                fastk_period=7, slowk_period=3, slowk_matype=0, 
                                                slowd_period=3, slowd_matype=0)
    df['CCI'] = talib.CCI(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=7)
    df['UO'] = talib.ULTOSC(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod1=3, timeperiod2=7, timeperiod3=14)
    df['WILLIAMS_R'] = talib.WILLR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=7)
    return df

def add_volatility_indicators(df):
    """
    Add volatility indicators to the DataFrame.
    """
    df['BB_HIGH'], df['BB_MAVG'], df['BB_LOW'] = talib.BBANDS(df['CLOSE'], timeperiod=10, nbdevup=3, nbdevdn=3, matype=0)
    df['ATR'] = talib.ATR(df['HIGH'], df['LOW'], df['CLOSE'], timeperiod=7)
    df['KC_HIGH'] = talib.SMA(df['CLOSE'], timeperiod=10) + 1.5 * df['ATR']
    df['KC_LOW'] = talib.SMA(df['CLOSE'], timeperiod=10) - 1.5 * df['ATR']
    df['DC_HIGH'] = df['HIGH'].rolling(window=10).max()
    df['DC_LOW'] = df['LOW'].rolling(window=10).min()
    df['DC_MAVG'] = (df['DC_HIGH'] + df['DC_LOW']) / 2
    df['PPO'] = talib.PPO(df['CLOSE'], fastperiod=6, slowperiod=13, matype=0)
    return df

def add_volume_indicators(df):
    """
    Add volume-based indicators to the DataFrame if volume is present.
    """
    df['OBV'] = talib.OBV(df['CLOSE'], df['VOLUME'])
    df['ADI'] = talib.AD(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'])
    df['CMF'] = talib.ADOSC(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'], fastperiod=3, slowperiod=10)
    df['FORCE_INDEX'] = df['CLOSE'].diff(1) * df['VOLUME']
    df['MFI'] = talib.MFI(df['HIGH'], df['LOW'], df['CLOSE'], df['VOLUME'], timeperiod=7)
    df['EOM'] = (df['HIGH'] - df['LOW']) / df['VOLUME']
    df['VPT'] = (df['CLOSE'].pct_change() * df['VOLUME']).cumsum()
    df['NVI'] = df['VOLUME'].pct_change().apply(lambda x: 0 if x > 0 else 1).cumsum()
    return df

def add_categorical_indicators(df):
    """
    Add categorical indicators to the DataFrame.
    """
    df['BULLISH'] = df['CLOSE'] > df['MA_5']
    df['BEARISH'] = df['CLOSE'] < df['MA_5']
    df['OVERBOUGHT_RSI'] = df['RSI'] > 70
    df['OVERSOLD_RSI'] = df['RSI'] < 30
    df['MACD_BULLISH'] = df['MACD'] > df['MACD_SIGNAL']
    df['MACD_BEARISH'] = df['MACD'] < df['MACD_SIGNAL']
    df['BB_BANDWIDTH_HIGH'] = (df['BB_HIGH'] - df['BB_LOW']) > (df['BB_HIGH'] - df['BB_LOW']).rolling(window=10).mean()
    df['BB_BANDWIDTH_LOW'] = (df['BB_HIGH'] - df['BB_LOW']) < (df['BB_HIGH'] - df['BB_LOW']).rolling(window=10).mean()
    df['STOCH_OVERBOUGHT'] = df['STOCH_K'] > 80
    df['STOCH_OVERSOLD'] = df['STOCH_K'] < 20
    df['UO_OVERBOUGHT'] = df['UO'] > 70
    df['UO_OVERSOLD'] = df['UO'] < 30
    return df

def add_candlestick_patterns(df):
    """
    Add candlestick pattern recognition features to the DataFrame as categorical variables.
    """
    patterns = {
        'CDL_DOJI': talib.CDLDOJI,
        'CDL_ENGULFING': talib.CDLENGULFING,
        'CDL_HAMMER': talib.CDLHAMMER,
        'CDL_SHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
        'CDL_MORNINGSTAR': lambda o, h, l, c: talib.CDLMORNINGSTAR(o, h, l, c, penetration=0.3),
        'CDL_EVENINGSTAR': lambda o, h, l, c: talib.CDLEVENINGSTAR(o, h, l, c, penetration=0.3),
        'CDL_HARAMI': talib.CDLHARAMI,
        'CDL_PIERCING': talib.CDLPIERCING,
        'CDL_DARKCLOUDCOVER': lambda o, h, l, c: talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0.3),
        'CDL_THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDL_THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
        'CDL_HANGINGMAN': talib.CDLHANGINGMAN,
        'CDL_INVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDL_BELTHOLD': talib.CDLBELTHOLD,
        'CDL_HARAMICROSS': talib.CDLHARAMICROSS,
        'CDL_KICKING': talib.CDLKICKING
    }
    
    for pattern_name, pattern_function in patterns.items():
        df[pattern_name] = pattern_function(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df[pattern_name] = df[pattern_name].apply(lambda x: int(x)).astype('category')
    
    return df

def add_volatility_features(df):
    """
    Add volatility-based features to the DataFrame.
    """
    df['VOLATILITY'] = df['CLOSE'].rolling(window=7).std()
    df['VOLATILITY_ZSCORE'] = (df['VOLATILITY'] - df['VOLATILITY'].rolling(window=30).mean()) / df['VOLATILITY'].rolling(window=30).std()
    return df

def add_technical_indicators(df, use_volume=True):
    """
    Main function to add all technical indicators and features to the DataFrame.
    """
    #df = add_moving_averages(df)
    #df = add_momentum_indicators(df)
    #df = add_volatility_indicators(df)
    
    #if use_volume:
        #df = add_volume_indicators(df)
    
    #df = add_categorical_indicators(df)
    #df = add_candlestick_patterns(df)
    df = add_volatility_features(df)
    
    return df



def wavelet_decomposition(df, wavelet='db1', level=1):
    """
    Decompose the time series using wavelet decomposition.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series.
    wavelet (str): The type of wavelet to use.
    level (int): The level of decomposition.
    
    Returns:
    coeffs: The wavelet coefficients.
    """
    coeffs = pywt.wavedec(df['CLOSE'], wavelet, level=level)
    coeffs_df = pd.DataFrame(coeffs).T
    coeffs_df.columns = [f'Wavelet_Coeff_Level_{i}' for i in range(len(coeffs))]
    
    return coeffs_df

import numpy as np
import matplotlib.pyplot as plt

def fourier_decomposition(df, n_harmonics=10):
    """
    Decompose the time series using Fourier decomposition.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series.
    n_harmonics (int): Number of Fourier harmonics to include.
    
    Returns:
    fourier_series: The reconstructed series using Fourier harmonics.
    """
    n = len(df)
    t = np.arange(n)
    f = np.fft.fftfreq(n)
    y_fft = np.fft.fft(df['CLOSE'])

    fourier_series = np.zeros(n)
    for i in range(1, n_harmonics + 1):
        fourier_series += np.real(y_fft[i]) * np.cos(2 * np.pi * f[i] * t) - np.imag(y_fft[i]) * np.sin(2 * np.pi * f[i] * t)
    
    return fourier_series


def get_all_featured_dataframes(df, use_volume=True, freq=24, decomposition_method='wavelet'):
    
    dataframes_dict = {
        'moving_averages': add_moving_averages(df.copy()),
        'momentum_indicators': add_momentum_indicators(df.copy()),
        'volatility_indicators': add_volatility_indicators(df.copy()),
        #'categorical_indicators': add_categorical_indicators(df.copy()),
        'candlestick_patterns': add_candlestick_patterns(df.copy()),
        'volatility_features': add_volatility_features(df.copy()),
    }
        
    #if decomposition_method == 'wavelet':
        #wavelet_coeffs = wavelet_decomposition(df, level=5)
        #dataframes_dict['wavelet_decomposition'] = wavelet_coeffs
    
    if use_volume: 
        dataframes_dict['volume_indicators'] = add_volume_indicators(df.copy())
    
    return dataframes_dict


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
