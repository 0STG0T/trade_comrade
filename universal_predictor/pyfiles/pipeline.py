import os
import requests
from predict import ModelPrediction
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class BuySellHoldPipeline:
    
    def __init__(self) -> None:
        self.cfg = self.load_config()
        self.model = ModelPrediction(model_path='./weis/cb')  # Initialize the model here
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_channel_id = os.getenv('TELEGRAM_CHANNEL_ID')

    def load_config(self):
        cfg = {
            'symbol': os.getenv('SYMBOL'),
            'interval': int(os.getenv('INTERVAL')),
            'tss_n_splits': int(os.getenv('TSS_N_SPLITS')),
            'n_back_features': int(os.getenv('N_BACK_FEATURES')),
            'tss_test_size': int(os.getenv('TSS_TEST_SIZE'))
        }
        return cfg

    def send_telegram_message(self, message):
        if self.telegram_token and self.telegram_channel_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_channel_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("Message sent successfully")
            else:
                print(f"Failed to send message: {response.text}")

    def start_prediction(self):
        print(f'Starting prediction for {self.cfg["symbol"]}!\n')
        buy_probas, hold_probas, sell_probas = self.model.full_prediction_cycle(
            symbol=self.cfg['symbol'],
            interval=self.cfg['interval'],
            n_back_features=self.cfg['n_back_features'],
            tss_n_splits=self.cfg['tss_n_splits'],
            tss_test_size=self.cfg['tss_test_size']
        )
        self.check_and_send_message(buy_probas, sell_probas, hold_probas)

    def predict(self, end_date):
        buy_probas, hold_probas, sell_probas = self.model.full_prediction_cycle(
            symbol=self.cfg['symbol'],
            interval=self.cfg['interval'],
            n_back_features=self.cfg['n_back_features'],
            tss_n_splits=self.cfg['tss_n_splits'],
            tss_test_size=self.cfg['tss_test_size'],
            end_date=end_date
        )
        self.check_and_send_message(buy_probas, sell_probas, hold_probas)

    def check_and_send_message(self, buy_probas, sell_probas, hold_probas):
        if buy_probas > 0.6 or sell_probas > 0.6:
            prediction_message = f'Predictions for {self.cfg["symbol"]}:\n'
            if buy_probas > 0.6:
                prediction_message += f'Buy probability: {buy_probas}\n'
            if sell_probas > 0.6:
                prediction_message += f'Sell probability: {sell_probas}\n'
            prediction_message += f'Hold probability: {hold_probas}'
            print(prediction_message)
            self.send_telegram_message(prediction_message)

    def predict_wrapper(self, interval):
        now_dt = datetime.now(ZoneInfo('Europe/Moscow'))
        closest_dt = self.get_closest_future_datetime(now_dt=now_dt, interval=interval)
        self.predict(closest_dt)

    @staticmethod
    def get_closest_future_datetime(now_dt, interval):
        delta_minutes = interval - (now_dt.minute % interval)
        if delta_minutes == 0:
            delta_minutes = interval  # Schedule to the next interval

        next_interval_time = now_dt + timedelta(minutes=delta_minutes)
        next_interval_time = next_interval_time.replace(second=0, microsecond=0)
        return next_interval_time

if __name__ == '__main__':
    pipe = BuySellHoldPipeline()
    pipe.start_prediction()
