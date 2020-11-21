import site
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

site.addsitedir('./model/')
from split import split
from normalize_data import normalize_data
from linear_regression import Linear_regression

if __name__ == '__main__':
    path = './data/Admission_Predict.csv'
    df = pd.read_csv(path, index_col=False)
    del df[df.columns[0]]
    #Add bias column to the dataframe
    bias = pd.Series(np.ones(len(df)), dtype=np.float64, name='bias')
    df = pd.concat([bias, df], axis=1, join='inner')    
    
    df = normalize_data(df, start=1, end=len(df.columns) - 2, method='rescale')
    print(df)
    X_train, Y_train, X_test, Y_test = split(df, rate = 3 / 4)

    w = np.zeros(len(X_train[0]))
    learning_rate = 0.06
    threhold = 1e-10
    gamma = 0.9

    Linear_model = Linear_regression(X_train, Y_train)

    #Variant of gradient descent
    w_bgd, loop_bgd, cost_bgd = Linear_model.fit('Minibatch', w, learning_rate, threhold, batch_size=len(X_train))
    print("Evaluate for batch gradient descent: ", Linear_model.evaluate(X_test, Y_test))

    w_sgd, loop_sgd, cost_sgd = Linear_model.fit('Minibatch', w,  learning_rate, threhold, batch_size=1)
    print("Evaluate for SGD: ", Linear_model.evaluate(X_test, Y_test))

    w_mnb, loop_mnb, cost_mnb = Linear_model.fit('Minibatch', w, learning_rate, threhold, batch_size=20)
    print("Evaluate for minibatch: ", Linear_model.evaluate(X_test, Y_test))

    #Test by using the psedo-inverse matrix
    print("Using batch gradient descent: \n\t w = ", w_mnb[:, 0])

    w1 = np.linalg.pinv(X_train).dot(Y_train)
    np.linalg.norm(w1 - w_mnb) / len(w_mnb)
    print("Using psedo-inverse matrix: \n\t w = ", w1[:, 0])

    #Using sklearn
    from sklearn import linear_model 
    a = linear_model.LinearRegression()
    a.fit(X_train, Y_train)
    print("Using sklearn: \n\t w = ", a.coef_[0, :])

    #Visualize 3 method
    fig, ax = plt.subplots()
    ax.plot(cost_sgd, label='SGD', color='r')
    ax.plot(cost_bgd, label='Batch GD', color='g')
    ax.plot(cost_mnb, label='Minibatch', color='orange')
    ax.set_xlim(0, 20)
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.show()



    #Use other methods
    w_nag, loop_nag, cost_nag = Linear_model.fit('NAG', w, learning_rate, threhold, 20, gamma)
    w_mmt, loop_mmt, cost_mmt = Linear_model.fit('Momentum',w, learning_rate, threhold, 20, gamma)
    w_rms, loop_rms, cost_rms = Linear_model.fit('RMSprop', w, learning_rate, threhold, 20, gamma)

    #Plot
    fig, ax = plt.subplots()
    ax.plot(cost_nag, label='NAG', color='yellow')
    ax.plot(cost_mnb, label='Minibatch', color='red')
    ax.plot(cost_mmt, label='Momentum', color='black')
    ax.plot(cost_rms, label='RMSprop', color='green')
    ax.set_xlim(0, 40)
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    plt.show()
