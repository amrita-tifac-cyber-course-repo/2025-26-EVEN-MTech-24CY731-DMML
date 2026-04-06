import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)

    # Remove leakage columns
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)

    # Encode categorical
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler