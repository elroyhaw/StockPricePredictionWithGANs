import yfinance as yf

ticker = yf.Ticker('AAPL')
ticker.get_balance_sheet()
