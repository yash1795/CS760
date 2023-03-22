from sre_constants import IN
import numpy as np
import pandas as pd
import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
import matplotlib.pyplot as plt

def init_params():
    W1 = np.random.uniform(-1,1,[300,784]) 
    W2 = np.random.uniform(-1,1,[200, 300]) 
    W3 = np.random.uniform(-1,1,[10, 200])
    Z2 = np.random.rand(200, 1)
    Z3 = np.random.rand(200, 1)
    return W1, W2, W3

def sigmoid (x):
    z = 1/(1 + np.exp(-x))
    return z

def softmax(x):
    a = np.exp(x)/np.exp(x).sum()
    return a

def forward (In, W1, W2, W3):  
    Y1 = np.matmul(W1,In)
    Z1 = sigmoid(Y1)
    Y2 = np.matmul (W2, Z1)
    Z2 = sigmoid(Y2)
    Y3 = np.matmul(W3, Z2)
    Z3 = softmax(Y3) 
    return Z1, Z2, Z3, Y1, Y2, Y3

def softmax_deriv(y_pred, y_actual):
    return y_pred - y_actual

def sigmoid_deriv(x) :
    return np.diag(x*(1-x))

def one_hot(Y):
    Y_hot = np.atleast_2d(np.zeros(10))
    Y_hot[0][Y] = 1
    return Y_hot.T

def back_prop (W2, W3 , Z1,Z2, Z3, In, Y):
    #print("back_prop : ")

    #print(np.shape(Z3))
    #print(np.shape(Z3))
    #print(np.shape(Z3))
    q = one_hot(Y)
    Z2 = np.atleast_2d(np.array(Z2))
    #print(np.shape(Z2))
    #print(np.shape(q))
    dW3 = np.matmul(Z3 - one_hot(Y), Z2.T)
    Z2_deriv = sigmoid_deriv(Z2.squeeze())
    A = np.matmul (Z2_deriv,W3.T)

    B = np.matmul(A,(Z3-one_hot(Y)))
    dW2 = np.matmul(B,Z1.T)
    Z1_deriv = sigmoid_deriv(Z1.squeeze())
    C = np.matmul (Z1_deriv, W2.T)
    D = np.matmul ( C, B)
    dW1 = np.matmul(D, In.T)
    return dW1, dW2, dW3

def update_params(W1, W2, W3, dW1, dW2, dW3, alpha):
    W1 = W1 - alpha * dW1  
    W2 = W2 - alpha * dW2  
    W3 = W3 - alpha * dW3    
    return W1, W2, W3

def get_predictions(Z3):
    return np.argmax(Z3, 0)


def gradient_descent(X, Y, alpha, iterations):
    W1, W2, W3 = init_params()
    for i in range(iterations):
        for j in range(5000):
            X = np.atleast_2d(np.array(X))
            X = X.T
            Z1, Z2, Z3, Y1, Y2, Y3 = forward(X, W1, W2, W3)
            dW1, dW2, dW3 = back_prop( W2, W3, Z1, Z2, Z3, X, Y)
            W1, W2, W3 = update_params(W1, W2, W3, dW1, dW2, dW3, alpha)
            y_cap = np.matmul(one_hot(Y).T,Z3)
            train_features, Y = mnist_data_train[j+1]
            train_flatten = torch.flatten(train_features.squeeze())
            X = train_flatten.numpy()
            predictions = get_predictions(Z3)  
    #plt.plot(Loss)
    return W1, W2, W3

mnist_data_train = torchvision.datasets.MNIST('.', train=True,download=True, transform=ToTensor())
mnist_data_test = torchvision.datasets.MNIST('.', train=False,download=True)
train_features, Y = mnist_data_train[0]
train_flatten = torch.flatten(train_features.squeeze())
In = train_flatten.numpy()

W1, W2, W3 = gradient_descent(In, Y, 0.001, 20)
count = 0
for i in range(10000):
    test_features, Y_test_actual = mnist_data_train[i]
    test_flatten = torch.flatten(test_features.squeeze())
    test_features = test_flatten.numpy()
    test_features = np.atleast_2d(np.array(test_features))
    test_features = test_features.T
    #print(np.shape(test_features))
    Z1, Z2, Z3, Y1, Y2, Y3 = forward (test_features, W1,W2,W3)
    Y_pred = get_predictions(Z3)
    if (Y_test_actual == Y_pred ):
        count = count + 1
Accuracy = count / 10000
print("Accuracy : ", Accuracy)