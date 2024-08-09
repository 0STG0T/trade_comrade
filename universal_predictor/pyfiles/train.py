from utils.utils import *
import pandas as pd
import numpy as np
from parsing.hist_parsing import KlineDataRetriever
from scipy.signal import argrelextrema
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
import time

warnings.filterwarnings('ignore')

print(os.listdir('./'))

# Load configuration from environment variables
symbol = os.getenv('SYMBOL')
interval = int(os.getenv('INTERVAL'))
tss_n_splits = int(os.getenv('TSS_N_SPLITS'))
n_back_features = int(os.getenv('N_BACK_FEATURES'))
tss_test_size = int(os.getenv('TSS_TEST_SIZE'))
target_window_size = int(os.getenv('TARGET_WINDOW_SIZE'))

parsing_params = {
    'category': 'linear',
    'symbol': symbol,
    'interval': interval,
    'testnet': False,
    'start_date': datetime(2024, 3, 1),
    'end_date': datetime.now()
}
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

def parse_and_preprocess_data(parsing_params, prediction=False) -> pd.DataFrame:
    retriever = KlineDataRetriever(category=parsing_params['category'], symbol=parsing_params['symbol'], 
                                interval=parsing_params['interval'], testnet=parsing_params['testnet'])
    
    if prediction:
        data = retriever.fetch_last_data(parsing_params['end_date'])
        
    else:
        data = retriever.fetch_data(parsing_params['start_date'], parsing_params['end_date'])
    
    preprocessed_data = preprocess_data(data)
    data_with_indicators = add_technical_indicators(preprocessed_data)
    data_with_holidays = add_holidays(data_with_indicators)
    final_data = drop_nans(data_with_holidays)
    
    # Apply the extrema target function
    final_data = add_extrema_targets(final_data, window_size=target_window_size)
    
    #print(f'Final data with length of {len(final_data)} preprocesed!')
    
    return final_data

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

def preprocess_for_training(final_data: pd.DataFrame):
    
    #final_data = balance_classes(final_data, 'TARGET')
    
    cols2drop = [] #['TICKER', 'PERIOD', 'DATETIME'
    
    today = datetime.now().day
    tomonth = datetime.now().month
    toyear = datetime.now().year
    
    #train_data = final_data.loc[final_data['DATETIME'] < f'{toyear}-{tomonth}-{today}'].drop(columns=cols2drop, errors='ignore')
    #test_data  = final_data.loc[final_data['DATETIME'] >= f'{toyear}-{tomonth}-{today}'].drop(columns=cols2drop, errors='ignore')

    train_data = final_data.copy()#.loc[:int(len(final_data)*0.95)].reset_index(drop=True)
    #test_data = final_data.loc[int(len(final_data)*0.95):].reset_index(drop=True)
    
    train_data = create_n_features(train_data, n=n_back_features)
    #test_data = create_n_features(test_data, n=n_back_features)
    
    #print(f'Train data len: {len(train_data)}, Val data len: {len(test_data)}')
    
    X, y = train_data.drop(columns=['TARGET', 'DATETIME']), train_data['TARGET']
    
    categorical_starts =[
        'TARGET',
        'BULLISH',
        'BEARISH',
        'OVERBOUGHT_RSI',
        'OVERSOLD_RSI',
        'MACD_BULLISH',
        'MACD_BEARISH',
        'BB_BANDWIDTH_HIGH',
        'BB_BANDWIDTH_LOW',
        'STOCH_OVERBOUGHT',
        'STOCH_OVERSOLD',
        'TSI_BULLISH',
        'TSI_BEARISH',
        'UO_OVERBOUGHT',
        'UO_OVERSOLD'
    ]

    cat_features = []

    for cs in categorical_starts:
        cat_features += [x if x.startswith(cs) else '' for x in X.columns]

    while '' in cat_features:
        cat_features.remove('')

    #return X, y, test_data, cat_features
    
    return X, y, cat_features

def train_model(X, y, cat_features, test_data=None):
    kf = TimeSeriesSplit(n_splits=tss_n_splits, test_size=tss_test_size)
    
    
    models_list = []
    
    print(X.shape)

    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = CatBoostClassifier(10000, l2_leaf_reg=2.5, bootstrap_type='Bayesian', early_stopping_rounds=40, use_best_model=True,
                                colsample_bylevel=0.85, grow_policy='Depthwise', random_state=42,
                                leaf_estimation_method='Newton', cat_features=cat_features, auto_class_weights='Balanced')
        
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
        
        score = f1_score(y_val, clf.predict(X_val).flatten(), average="weighted")
        
        print(f'FOLD {i} F1: {score}')
        
        models_list.append((clf, score))
        
    model = max(models_list, key=lambda x: x[1])[0]
    
    print(f'Best model F1: {max(models_list, key=lambda x: x[1])[1]}')
    
    """buy_probas = model.predict_proba(test_data.drop(columns=['TARGET', 'DATETIME']))[:, 0]
    hold_probas = model.predict_proba(test_data.drop(columns=['TARGET', 'DATETIME']))[:, 1]
    sell_probas = model.predict_proba(test_data.drop(columns=['TARGET', 'DATETIME']))[:, 2]"""
    
    return model

def train_and_save() -> str:
    print('Starting parsing!')
    t1 = time.time()
    final_data = parse_and_preprocess_data(parsing_params=parsing_params)
    print('Parsed in', time.time() - t1, 'seconds')
    
    #X, y, test_data, cat_features = preprocess_for_training(final_data=final_data)
    X, y,  cat_features = preprocess_for_training(final_data=final_data)
    model = train_model(X=X, y=y, test_data=None, cat_features=cat_features)
    
    model.save_model('./weis/cb')
    
    print('Model saved!')
    
    return model

if __name__ == '__main__':
    train_and_save()