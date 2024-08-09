from datetime import datetime, timedelta
import pandas as pd
from catboost import CatBoostClassifier
from train import parse_and_preprocess_data, create_n_features
import warnings
import zoneinfo

warnings.filterwarnings('ignore')

class ModelPrediction:
    def __init__(self, model_path, verbose=False):
        self.verbose = verbose
        self.model = self.load_model(model_path)
    
    @staticmethod
    def yesterday(end_dt=None):
        if not end_dt:
            return datetime.now(zoneinfo.ZoneInfo('Europe/Moscow')) - timedelta(days=3)
        return end_dt - timedelta(days=4)
    
    @staticmethod
    def load_model(model_path):
        return CatBoostClassifier().load_model(model_path)
    
    @staticmethod
    def preprocess_for_prediction(final_data: pd.DataFrame, n_back_features):
        final_data = create_n_features(final_data, n=n_back_features)
        X = final_data.drop(columns=['TARGET', 'DATETIME'])
        return X
    
    def predict(self, X):
        buy_probas = self.model.predict_proba(X)[-1, 0]
        hold_probas = self.model.predict_proba(X)[-1, 1]
        sell_probas = self.model.predict_proba(X)[-1, 2]
        return buy_probas, hold_probas, sell_probas
    
    def full_prediction_cycle(self, symbol, interval, n_back_features, tss_n_splits, tss_test_size, start_date=None, end_date=None):
        parsing_params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'testnet': False,
            'start_date': start_date,
            'end_date': end_date
        }
        
        final_data = parse_and_preprocess_data(parsing_params=parsing_params, prediction=True)
        X = self.preprocess_for_prediction(final_data=final_data, n_back_features=n_back_features)
        
        buy_probas, hold_probas, sell_probas = self.predict(X=X)
        
        probas_string = f'Buy probability: {buy_probas}\nHold probability: {hold_probas}\nSell probability: {sell_probas}'
        datetime_string = f'Predictions for {final_data["DATETIME"].iloc[-1]}:'
        
        if self.verbose:
            print(datetime_string)
            print(probas_string)
            print('\n')

        return buy_probas, hold_probas, sell_probas
