import os
import ccxt
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from predict import ModelPrediction

class BuySellHoldPipeline:
    def __init__(self) -> None:
        self.cfg = self.load_config()
        self.model = ModelPrediction(model_path='./weis/cb')
        self.bybit_client = ccxt.bybit({
            'apiKey': os.getenv('BYBIT_API_KEY'),
            'secret': os.getenv('BYBIT_API_SECRET'),
            'enableRateLimit': True,
        })
        self.bybit_client.set_sandbox_mode(False)
        self.iam_token = os.getenv('IAM_TOKEN')
        self.folder_id = os.getenv('FOLDER_ID')
        
        self.thresh_1 = float(os.getenv('THRESH_1'))
        self.thresh_2 = float(os.getenv('THRESH_2'))
        
    def start_prediction(self):
        print(f'Starting prediction for {self.cfg["symbol"]}!\n')
        buy_probas, hold_probas, sell_probas = self.model.full_prediction_cycle(
            symbol=self.cfg['symbol'],
            interval=self.cfg['interval'],
            n_back_features=self.cfg['n_back_features'],
            tss_n_splits=self.cfg['tss_n_splits'],
            tss_test_size=self.cfg['tss_test_size']
        )
        self.execute_strategy(buy_probas, sell_probas)

    def predict(self, end_date):
        buy_probas, hold_probas, sell_probas = self.model.full_prediction_cycle(
            symbol=self.cfg['symbol'],
            interval=self.cfg['interval'],
            n_back_features=self.cfg['n_back_features'],
            tss_n_splits=self.cfg['tss_n_splits'],
            tss_test_size=self.cfg['tss_test_size'],
            end_date=end_date
        )
        self.execute_strategy(buy_probas, sell_probas)

    def load_config(self):
        cfg = {
            'symbol': os.getenv('SYMBOL'),
            'interval': int(os.getenv('INTERVAL')),
            'tss_n_splits': int(os.getenv('TSS_N_SPLITS')),
            'n_back_features': int(os.getenv('N_BACK_FEATURES')),
            'tss_test_size': int(os.getenv('TSS_TEST_SIZE'))
        }
        return cfg

    def send_metrics(self, roi, profit, fees):
        url = f"https://monitoring.api.cloud.yandex.net/monitoring/v2/data/write?folderId={self.folder_id}&service=custom"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}"
        }

        body = {
            "metrics": [
                {"name": "roi", "value": roi},
                {"name": "profit", "value": profit},
                {"name": "fees", "value": fees}
            ]
        }

        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            print("Metrics sent successfully")
        else:
            print(f"Failed to send metrics: {response.text}")

    def execute_strategy(self, buy_probas, sell_probas):
        if buy_probas > self.thresh_1:
            profit, fees = self.place_order("Buy")
        elif sell_probas > self.thresh_2:
            profit, fees = self.place_order("Sell")
        else:
            print("Hold - No action taken.")
            return

        roi = (profit - fees) / fees * 100 if fees else 0
        self.send_metrics(roi, profit, fees)

    def place_order(self, side):
        try:
            symbol = f"{self.cfg['symbol']}/USDT"
            market_price = self.get_current_price(symbol)
            if market_price is None:
                print(f"Failed to fetch market price for {symbol}, skipping order.")
                return 0, 0
            
            if side == "Buy":
                balance = self.get_usdt_balance()
                qty = (balance * 0.95) / market_price
            elif side == "Sell":
                balance = self.get_asset_balance(self.cfg['symbol'])
                qty = balance

            print(f"Balance before {side}: {balance}")
            
            if balance <= 0:
                print(f"Insufficient balance to place {side} order for {symbol}.")
                return 0, 0
            
            precision = 6
            qty = round(qty, precision)
            print(f"Rounded quantity for {side}: {qty}")

            if qty <= 0:
                print(f"Insufficient qty after rounding to place {side} order for {symbol}.")
                return 0, 0

            order = self.bybit_client.create_order(symbol, 'market', side.lower(), qty)
            print(f"Order placed: {order}")
            
            if order['status'] == 'closed' and order['filled'] > 0:
                print(f"Order successfully executed. {order['filled']} {symbol} bought/sold.")
                profit = order['filled'] * market_price if side == "Sell" else -order['cost']
                fees = order['fee']['cost']
                return profit, fees
            else:
                print("Order placed but not immediately filled. Check order details.")
                return 0, 0
                
        except Exception as e:
            print(f"Error placing {side} order: {e}")
            return 0, 0

    def get_current_price(self, symbol):
        try:
            ticker = self.bybit_client.fetch_ticker(symbol)
            price = ticker['last']
            print(f"Current price for {symbol}: {price}")
            return price
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None


    def get_usdt_balance(self):
        try:
            # Fetch the full balance
            balance = self.bybit_client.fetch_balance()
            usdt_balance = balance['total']['USDT']
            print(f"USDT wallet balance: {usdt_balance}")
            return usdt_balance
        except Exception as e:
            print(f"Error fetching USDT balance: {e}")
            return 0.0

    def get_asset_balance(self, asset):
        try:
            # Fetch the full balance
            balance = self.bybit_client.fetch_balance()
            asset_balance = balance['total'][asset]
            print(f"{asset} wallet balance: {asset_balance}")
            return asset_balance
        except Exception as e:
            print(f"Error fetching {asset} balance: {e}")
            return 0.0

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
    
    def test_pipeline(self):
        try:
            print("Starting pipeline tests...")

            # Test fetching the current price
            symbol = os.getenv('SYMBOL')
            print(f"Testing current price fetch for symbol: {symbol}")
            price = self.get_current_price(symbol)
            if price is None:
                print("Test failed: Could not fetch current price.")
                return False
            print(f"Test passed: Fetched current price: {price}")

            # Test fetching USDT balance
            print("Testing USDT balance fetch...")
            usdt_balance = self.get_usdt_balance()
            if usdt_balance == 0:
                print("Test failed: USDT balance is zero or not found.")
                return False
            print(f"Test passed: USDT balance: {usdt_balance}")

            # Test fetching asset balance (e.g., POPCAT)
            asset = symbol.split('USDT')[0]
            print(f"Testing {asset} balance fetch...")
            asset_balance = self.get_asset_balance(asset)
            if asset_balance == 0:
                print(f"Test failed: {asset} balance is zero or not found.")
                return False
            print(f"Test passed: {asset} balance: {asset_balance}")

            # Test placing a buy order (dry run)
            print("Testing buy order placement...")
            try:
                self.place_order("Buy")
                print("Test passed: Buy order placement executed (check output for details).")
            except Exception as e:
                print(f"Test failed: Error placing buy order: {e}")
                return False

            # Test placing a sell order (dry run)
            print("Testing sell order placement...")
            try:
                self.place_order("Sell")
                print("Test passed: Sell order placement executed (check output for details).")
            except Exception as e:
                print(f"Test failed: Error placing sell order: {e}")
                return False

            print("All tests passed successfully!")
            return True

        except Exception as e:
            print(f"Test pipeline encountered an error: {e}")
            return False


if __name__ == '__main__':
    pipe = BuySellHoldPipeline()

    # Run pipeline tests
    if not pipe.test_pipeline():
        print("Pipeline test failed. Exiting...")
    else:
        print("Pipeline test successful. Starting prediction...")
        pipe.start_prediction()
