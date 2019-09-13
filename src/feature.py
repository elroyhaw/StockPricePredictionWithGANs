import pandas as pd


def get_technical_indicators(df):
    # Create 7 and 21 days Moving Average
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()

    # Create MACD
    df['26EMA'] = df['Close'].ewm(span=26).mean()
    df['12EMA'] = df['Close'].ewm(span=12).mean()
    df['MACD'] = (df['12EMA'] - df['26EMA'])

    # Create Bollinger Bands
    df['20SD'] = df['Close'].rolling(window=20).std()
    df['UPPER_BAND'] = df['MA21'] + (df['20SD'] * 2)
    df['LOWER_BAND'] = df['MA21'] - (df['20SD'] * 2)

    # Create Exponential moving average
    df['EMA'] = df['Close'].ewm(com=0.5).mean()

    # Create Momentum
    df['MOMENTUM'] = df['Close'] - 1

    return df


def get_all_features(df):
    tech_df = get_technical_indicators(df)
    return pd.concat([df, tech_df], axis=1)
