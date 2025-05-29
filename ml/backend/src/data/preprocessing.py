from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    df = df.dropna()
    df = df[(df['volume'] > 0) & (df['price'] > 0)]
    return df

def add_technical_indicators(df):
    df['rsi'] = RSIIndicator(close=df['close']).rsi()
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    return df

def normalize_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler 