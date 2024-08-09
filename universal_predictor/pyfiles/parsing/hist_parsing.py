from datetime import datetime
import pandas as pd
from pybit.unified_trading import HTTP
import zoneinfo

class KlineDataRetriever:
    def __init__(self, category: str, symbol: str, interval: int, testnet: bool = False):
        self.category = category
        self.symbol = symbol
        self.interval = interval
        self.session = HTTP(testnet=testnet)
        self.df = pd.DataFrame()
        #print('Parser initialized!')
        
    def fetch_last_data(self, end_date: datetime = None, start_date: datetime = None):
        if not end_date:
            end_date = datetime.now(zoneinfo.ZoneInfo('Europe/Moscow'))
            
        end_timestamp = int(end_date.timestamp() * 1000)
        interval_ms = self.interval * 60 * 1000
        
        if start_date:
            start_timestamp = int(start_date.timestamp() * 1000) 
        else:
            start_timestamp = end_timestamp - interval_ms * 50
        
        resp = self.session.get_kline(
            category=self.category,
            symbol=self.symbol,
            interval=self.interval,
            start=start_timestamp,
            end=end_timestamp 
        )
        
        temp_df = pd.DataFrame.from_dict(resp['result']['list'])
        
        temp_df.columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        temp_df['DATETIME'] = pd.to_datetime(temp_df['start_time'].apply(lambda x: datetime.fromtimestamp(int(x)/1000, tz=zoneinfo.ZoneInfo('Europe/Moscow'))))
        temp_df = temp_df.sort_values(by='DATETIME', ascending=True).reset_index(drop=True).drop(columns=['turnover', 'start_time'])
        
        return temp_df

    def fetch_data(self, start_date: datetime, end_date: datetime = None):

        if end_date is None:
            datetime.now(zoneinfo.ZoneInfo('Europe/Moscow'))

        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        interval_ms = self.interval * 60 * 1000  # Convert interval to milliseconds
        current_left = end_timestamp - interval_ms * 200
        current_right = end_timestamp
        
        while current_left > start_timestamp:
            
            resp = self.session.get_kline(
                category=self.category,
                symbol=self.symbol,
                interval=self.interval,
                start=current_left,
                end=current_right 
            )
            
            current_right = current_right - interval_ms * 200
            current_left = current_left - interval_ms * 200
            
            temp_df = pd.DataFrame.from_dict(resp['result']['list'])
            if temp_df.empty:
                print('Empty')
                break  # Break the loop if no data is returned
                    
            temp_df.columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
            temp_df['DATETIME'] = pd.to_datetime(temp_df['start_time'].apply(lambda x: datetime.fromtimestamp(int(x)/1000, tz=zoneinfo.ZoneInfo('Europe/Moscow'))))
                    
            self.df = pd.concat([self.df, temp_df], ignore_index=True)

        # Sort DataFrame by DATETIME
        self.df = self.df.sort_values(by='DATETIME', ascending=True).reset_index(drop=True).drop(columns=['turnover', 'start_time'])
        
        #print('Fetched successfully!')
        
        return self.df


