import numpy as np
import random


class Linear_regression:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.w = np.zeros(len(X_train[0])).reshape(-1, 1)

    def loss_function(self, w):
        return np.sum((self.X_train.dot(w) - self.Y_train) ** 2) / (2 * len(self.X_train)) 

    def grad(self, w, batch):
        X_batch = self.X_train[batch[0] : batch[1]]
        Y_batch = self.Y_train[batch[0] : batch[1]]
        return X_batch.T.dot(X_batch.dot(w) - Y_batch) / (batch[1] - batch[0])

    def fit(self, method, initialize, learning_rate, threhold, batch_size, gamma=None, max_loop=2000):
        epoch = 0
        cost = []
        w = np.array(initialize).reshape(-1, 1)
        len_interval = int(len(self.X_train) / batch_size)

        if method == 'Minibatch':
            while np.sum(np.abs(self.grad(w, [0 , len(self.X_train)]))) > threhold:
                cost.append(self.loss_function(w))
                l_index = random.randint(0, len_interval - 1) * batch_size
                h_index = l_index + batch_size
                w = w - learning_rate * self.grad(w, [l_index, h_index])
                epoch += 1
                if epoch > max_loop:
                    break

        elif method == 'Momentum':
            v = np.zeros(len(w)).reshape(-1, 1)
            while np.sum(np.abs(self.grad(w, [0, len(self.X_train)]))) > threhold:
                cost.append(self.loss_function(w))
                l_index = random.randint(0, len_interval - 1) * batch_size
                h_index = l_index + batch_size
                v = gamma * v + learning_rate * self.grad(w , [l_index, h_index])
                w -= v
                epoch += 1
                if epoch > max_loop:
                    break

        elif method == 'NAG':
            v = np.zeros(len(w)).reshape(-1, 1)
            while np.sum(np.abs(self.grad(w, [0, len(self.X_train)]))) > threhold:
                cost.append(self.loss_function(w))
                l_index = random.randint(0, len_interval - 1) * batch_size
                h_index = l_index + batch_size
                v = gamma * v + learning_rate * self.grad(w - gamma * v, [l_index, h_index])
                w -= v
                epoch += 1
                if epoch > max_loop:
                    break
            
        elif method == 'RMSprop':
            E = np.zeros(len(w)).reshape(-1, 1)
            epsilon = 1e-10
            while np.sum(np.abs(self.grad(w, [0, len(self.X_train)]))) > threhold:
                cost.append(self.loss_function(w))
                l_index = random.randint(0, len_interval - 1) * batch_size
                h_index = l_index + batch_size
                E = gamma * E + (1 - gamma) * self.grad(w, [l_index, h_index]) ** 2
                w -= learning_rate / np.sqrt(E + epsilon) * self.grad(w, [l_index, h_index])
                epoch += 1
                if epoch > max_loop:
                    break

        return w, epoch, cost

    def evaluate(self, X_test, Y_test):
        MAE = np.sum(np.abs(X_test.dot(self.w) - Y_test)) / len(X_test)
        MSE = np.sqrt(np.sum((X_test.dot(self.w) - Y_test) ** 2)) / len(X_test)
        VarTest = np.sum((Y_test - np.mean(Y_test) ** 2)) / len(X_test)
        R2 = 1 - MSE / VarTest
        return {'MAE' : np.round(MAE, decimals=4), 'MSE' : np.round(MSE, decimals=4), 'R2' : np.round(R2, decimals=4)}
