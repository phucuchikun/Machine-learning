import pandas as pd

def split(df, rate):
    #Suffle the dataframe
    df = df.sample(frac=1, random_state=7)

    train_num = int(len(df) * rate)
    X_train = df[:train_num].values[:, :-1]
    Y_train = df[:train_num].values[:, -1].reshape(-1, 1)
    X_test = df[train_num:].values[:, :-1]
    Y_test = df[train_num:].values[:, -1].reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test
