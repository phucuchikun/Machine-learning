import pandas as pd
import numpy as np

def normalize_data(df, start, end, method='rescale'):
    if method == 'rescale':
        for i in df.columns[start:end]:
            df[i] = (df[i] - np.min(df[i])) / (np.max(df[i]) - np.min(df[i]))
    elif method == 'zscore':
        for i in df.columns[start:end]:
            df[i] = (df[i] - np.mean(df[i])) / np.sqrt(np.sum((df[i] - np.mean(df[i])) ** 2))
    else:
        raise 'Method invalid.'
    return df
