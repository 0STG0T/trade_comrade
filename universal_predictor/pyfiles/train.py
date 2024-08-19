import os
import time
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support
from catboost import CatBoostClassifier
from parsing.hist_parsing import KlineDataRetriever
from utils.utils import *

warnings.filterwarnings('ignore')

# Load configuration from environment variables
symbol = os.getenv('SYMBOL')
interval = int(os.getenv('INTERVAL'))
tss_n_splits = int(os.getenv('TSS_N_SPLITS'))
n_back_features = int(os.getenv('N_BACK_FEATURES'))
tss_test_size = int(os.getenv('TSS_TEST_SIZE'))
target_window_size = int(os.getenv('TARGET_WINDOW_SIZE'))
actual_test_size = int(os.getenv('TEST_SIZE'))

# Parsing parameters for data retrieval
parsing_params = {
    'category': 'linear',
    'symbol': symbol,
    'interval': interval,
    'testnet': False,
    'start_date': datetime(2024, 1, 1),
    'end_date': datetime.now()
}

def parse_and_preprocess_data(parsing_params, prediction=False) -> pd.DataFrame:
    retriever = KlineDataRetriever(
        category=parsing_params['category'], 
        symbol=parsing_params['symbol'], 
        interval=parsing_params['interval'], 
        testnet=parsing_params['testnet']
    )
    
    # Fetch data based on mode (prediction or training)
    data = retriever.fetch_last_data(parsing_params['end_date']) if prediction else retriever.fetch_data(parsing_params['start_date'], parsing_params['end_date'])
    
    # Data preprocessing and feature engineering
    preprocessed_data = preprocess_data(data)
    data_with_indicators = add_technical_indicators(preprocessed_data)
    data_with_holidays = add_holidays(data_with_indicators)
    final_data = drop_nans(data_with_holidays)
    
    # Apply the extrema target function
    final_data = add_extrema_targets(final_data, window_size=target_window_size)
    
    return final_data

def preprocess_for_training(final_data: pd.DataFrame):
    # Split data into training and testing sets
    train_data = final_data[:-actual_test_size]
    test_data = final_data[-actual_test_size:]
    
    # Create lagged features
    train_data = create_n_features(train_data, n=n_back_features)
    test_data = create_n_features(test_data, n=n_back_features)
    
    # Separate features and target variables
    X, y = train_data.drop(columns=['TARGET', 'DATETIME']), train_data['TARGET']
    X_test, y_test = test_data.drop(columns=['TARGET', 'DATETIME']), test_data['TARGET']
    
    # Identify categorical features
    categorical_starts = [
        'TARGET', 'BULLISH', 'BEARISH', 'OVERBOUGHT_RSI', 'OVERSOLD_RSI',
        'MACD_BULLISH', 'MACD_BEARISH', 'BB_BANDWIDTH_HIGH', 'BB_BANDWIDTH_LOW',
        'STOCH_OVERBOUGHT', 'STOCH_OVERSOLD', 'TSI_BULLISH', 'TSI_BEARISH',
        'UO_OVERBOUGHT', 'UO_OVERSOLD'
    ]
    cat_features = [col for cs in categorical_starts for col in X.columns if col.startswith(cs)]
    
    return X, y, cat_features, X_test, y_test

def evaluate_model_performance(model, X_val, y_val, dataset_name):
    y_pred = model.predict(X_val).flatten()
    
    # Calculate and print evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average=None)
    f1_macro = f1_score(y_val, y_pred, average="macro")
    
    print(f'\nPerformance on {dataset_name}:')
    print(f'F1-macro: {f1_macro}')
    print(f'Precision per class: {precision}')
    print(f'Recall per class: {recall}')
    print(f'F1 per class: {f1}')
    return f1_macro

def train_model(X, y, cat_features, X_test, y_test):
    kf = TimeSeriesSplit(n_splits=tss_n_splits, test_size=tss_test_size)
    
    models_list = []
    
    print(f"Training on data of shape: {X.shape}")
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = CatBoostClassifier(
            iterations=10000, 
            l2_leaf_reg=1.5, 
            bootstrap_type='Bayesian', 
            early_stopping_rounds=100, 
            use_best_model=True,
            colsample_bylevel=0.95, 
            random_state=42, 
            posterior_sampling=True,
            leaf_estimation_method='Newton', 
            cat_features=cat_features, 
            auto_class_weights='Balanced'
        )
        
        clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=100)
        
        # Evaluate performance on the validation set
        f1_macro_val = evaluate_model_performance(clf, X_val, y_val, f"Validation Fold {i}")
        
        models_list.append((clf, f1_macro_val))
        
    # Select the best model based on validation F1-macro
    best_model = max(models_list, key=lambda x: x[1])[0]
    
    # Evaluate the best model on the test data
    evaluate_model_performance(best_model, X_test, y_test, "Test Set")
    
    return best_model

def train_and_save() -> str:
    print('Starting data parsing and preprocessing...')
    start_time = time.time()
    
    final_data = parse_and_preprocess_data(parsing_params=parsing_params)
    
    elapsed_time = time.time() - start_time
    print(f'Data parsed and preprocessed in {elapsed_time:.2f} seconds')
    
    # Prepare data for training
    X, y, cat_features, X_test, y_test = preprocess_for_training(final_data=final_data)
    
    # Train the model and evaluate
    model = train_model(X=X, y=y, cat_features=cat_features, X_test=X_test, y_test=y_test)
    
    # Save the trained model
    model_path = './weis/cb'
    model.save_model(model_path)
    
    print(f'Model saved to {model_path}!')
    
    return model_path

if __name__ == '__main__':
    train_and_save()
