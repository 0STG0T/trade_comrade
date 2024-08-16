from datetime import datetime, timedelta
import pandas as pd
from catboost import CatBoostClassifier
from train import parse_and_preprocess_data
from utils.utils import *
import warnings
import zoneinfo
import os

warnings.filterwarnings('ignore')

class ModelPrediction:
    def __init__(self, model_directory, verbose=False):
        self.verbose = verbose
        self.model_directory = model_directory
        self.models = self.load_all_models(model_directory)
    
    @staticmethod
    def yesterday(end_dt=None):
        if not end_dt:
            return datetime.now(zoneinfo.ZoneInfo('Europe/Moscow')) - timedelta(days=3)
        return end_dt - timedelta(days=4)
    
    def load_all_models(self, model_directory):
        models = {}
        for model_file in os.listdir(model_directory):
            if model_file.endswith('_model'):
                model_key = model_file.replace('_model', '')
                models[model_key] = CatBoostClassifier().load_model(os.path.join(model_directory, model_file))
        return models
    
    @staticmethod
    def preprocess_for_prediction(final_data: pd.DataFrame, n_back_features):
        final_data = create_n_features(final_data, n=n_back_features)
        X = final_data.drop(columns=['TARGET', 'DATETIME'])
        return X
    
    def predict(self, X, model):
        buy_probas = model.predict_proba(X)[-1, 0]
        hold_probas = model.predict_proba(X)[-1, 1]
        sell_probas = model.predict_proba(X)[-1, 2]
        return buy_probas, hold_probas, sell_probas
    
    def full_prediction_cycle(self, symbol, interval, n_back_features, start_date=None, end_date=None):
        parsing_params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'testnet': False,
            'start_date': start_date,
            'end_date': end_date
        }
        
        final_data_dict = parse_and_preprocess_data(parsing_params=parsing_params, prediction=True)
        aggregated_probas = {'buy': 0, 'hold': 0, 'sell': 0}
        
        for key, final_data in final_data_dict.items():
            X = self.preprocess_for_prediction(final_data=final_data, n_back_features=n_back_features)
            buy_probas, hold_probas, sell_probas = self.predict(X=X, model=self.models[key])
            
            # Aggregate the probabilities (simple average)
            aggregated_probas['buy'] += buy_probas
            aggregated_probas['hold'] += hold_probas
            aggregated_probas['sell'] += sell_probas
        
        # Calculate the average probability
        num_models = len(self.models)
        aggregated_probas = {k: v / num_models for k, v in aggregated_probas.items()}
        
        probas_string = (f'Buy probability: {aggregated_probas["buy"]}\n'
                         f'Hold probability: {aggregated_probas["hold"]}\n'
                         f'Sell probability: {aggregated_probas["sell"]}')
        datetime_string = f'Predictions for {final_data["DATETIME"].iloc[-1]}:'
        
        if self.verbose:
            print(datetime_string)
            print(probas_string)
            print('\n')

        return aggregated_probas['buy'], aggregated_probas['hold'], aggregated_probas['sell']
