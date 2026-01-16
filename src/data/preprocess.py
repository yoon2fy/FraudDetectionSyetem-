import numpy as np

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    # log transform
    for col in ['Amount', 'Time']:
        X[col] = np.log(X[col] + 1)

    # min-max scaling
    for col in X.columns:
        X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())

    return X, y
