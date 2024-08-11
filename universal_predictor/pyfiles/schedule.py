from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pipeline import BuySellHoldPipeline
import time
import os

def main():
    interval = int(os.getenv('INTERVAL'))

    zi = ZoneInfo('Europe/Moscow')
    pipeline = BuySellHoldPipeline()
    
    now_dt = datetime.now(zi)
    closest_dt = pipeline.get_closest_future_datetime(now_dt=now_dt, interval=interval)
    
    scheduler = BackgroundScheduler(timezone=zi)
    scheduler.add_job(
        pipeline.predict_wrapper, 
        IntervalTrigger(minutes=interval, start_date=closest_dt),
        args=[interval],
        id='buy_sell_hold_job',  # Optional: giving a name to the job
        replace_existing=True    # Optional: replaces existing job with the same id
    )
    scheduler.start()
    
    print(f'Scheduled prediction every {interval} minutes, starting from: {closest_dt}')
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down scheduler...")
        scheduler.shutdown()

if __name__ == '__main__':
    main()
