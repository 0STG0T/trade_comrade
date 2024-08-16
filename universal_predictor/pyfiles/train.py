from utils.utils import *
import pandas as pd
import numpy as np
from parsing.hist_parsing import KlineDataRetriever
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
actual_test_size = int(os.getenv('TEST_SIZE'))

parsing_params = {
    'category': 'linear',
    'symbol': symbol,
    'interval': interval,
    'testnet': False,
    'start_date': datetime(2024, 1, 1),
    'end_date': datetime.now()
}

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
    
    return final_data

def preprocess_for_training(final_data: pd.DataFrame):

    train_data = final_data[:-actual_test_size]
    test_data = final_data[-actual_test_size:]
    
    train_data = create_n_features(train_data, n=n_back_features)
    test_data = create_n_features(test_data, n=n_back_features)
    
    X, y = train_data.drop(columns=['TARGET', 'DATETIME']), train_data['TARGET']
    X_test, y_test = test_data.drop(columns=['TARGET', 'DATETIME']), test_data['TARGET']
    
    categorical_starts = [
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
        'UO_OVERBOUGHT',
        'UO_OVERSOLD',
        'CDL_DOJI',
        'CDL_ENGULFING',
        'CDL_HAMMER',
        'CDL_SHOOTINGSTAR',
        'CDL_MORNINGSTAR',
        'CDL_EVENINGSTAR',
        'CDL_HARAMI',
        'CDL_PIERCING',
        'CDL_DARKCLOUDCOVER',
        'CDL_THREE_WHITE_SOLDIERS',
        'CDL_THREE_BLACK_CROWS',
        'CDL_HANGINGMAN',
        'CDL_INVERTEDHAMMER',
        'CDL_BELTHOLD',
        'CDL_HARAMICROSS',
        'CDL_KICKING'
    ]

    cat_features = []

    for cs in categorical_starts:
        cat_features += [x if x.startswith(cs) else '' for x in X.columns]

    while '' in cat_features:
        cat_features.remove('')
    
    return X, y, cat_features, X_test, y_test

def train_model(X, y, cat_features, X_test, y_test):
    kf = TimeSeriesSplit(n_splits=tss_n_splits, test_size=tss_test_size)
    
    models_list = []
    
    print(X.shape)

    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = CatBoostClassifier(10000, l2_leaf_reg=1.5, bootstrap_type='Bayesian', early_stopping_rounds=100, use_best_model=True,
                                colsample_bylevel=0.95, random_state=42, posterior_sampling=True,
                                leaf_estimation_method='Newton', cat_features=cat_features, auto_class_weights='Balanced')
        
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
        
        score = f1_score(y_val, clf.predict(X_val).flatten(), average="weighted")
        
        print(f'FOLD {i} F1: {score}')
        
        models_list.append((clf, score))
        
    model = max(models_list, key=lambda x: x[1])[0]
    
    # Evaluating on the test data
    score = f1_score(y_test, clf.predict(X_test).flatten(), average="weighted")
    
    print(f'Test data F1: {score}')
    
    return model

def train_and_save() -> str:
    print('Starting parsing!')
    t1 = time.time()
    final_data = parse_and_preprocess_data(parsing_params=parsing_params)
    print('Parsed in', time.time() - t1, 'seconds')
    
    X, y, cat_features, X_test, y_test = preprocess_for_training(final_data=final_data)
    model = train_model(X=X, y=y, cat_features=cat_features, X_test=X_test, y_test=y_test)
    
    model.save_model('./weis/cb')
    
    print('Model saved!')
    
    return model

if __name__ == '__main__':
    train_and_save()