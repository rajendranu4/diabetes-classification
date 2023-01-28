import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Net_Diabetes import DiabetesNetwork


def data_preprocessing(data_file):
    df = pd.read_csv(data_file)
    print("Shape of the dataset")
    print(df.shape)

    # reducing 8d data to 2d data to visualize the clustering of class variables
    X_2d = TSNE(n_components=2).fit_transform(df.loc[:, df.columns != 'Outcome'])

    fig, ax = plt.subplots()
    df_dx = X_2d
    df_dy = np.asarray(df)[:, -1]
    ax.scatter(df_dx[df_dy == 0, 0], df_dx[df_dy == 0, 1], label="0")
    ax.scatter(df_dx[df_dy == 1, 0], df_dx[df_dy == 1, 1], label="1", color="r")
    sns.despine()
    ax.legend()

    # checking if there are any missing values in the dataset
    print("\nChecking if there are any missing values")
    print(df.isna().sum())

    # separating input variables as X and output class as y
    X = df.iloc[:, :8]
    y = df.iloc[:, 8]

    # splitting dataset into training and testing using sklearn preprocessing utility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    print("\nX - train")
    print(X_train.head())
    print("\ny - train")
    print(y_train.head())

    # normalizing - z_normalization the inputs for better performance ==> z_score = (x - mean(x)) / standard_deviation(x)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    print("\nNormalized data")
    print(X_train)
    print(X_test)

    # converting the inputs into tensor as torch supports tensor variables
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)

    print("\nTensor data")
    print(X_train)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    data_file = 'diabetes.csv'

    # parameters for the neural network
    hidden_neurons = 15
    learning_rate = 0.05
    epochs = 20000

    X_train, y_train, X_test, y_test = data_preprocessing(data_file)

    d_network = DiabetesNetwork(X_train.shape[1], hidden_neurons)

    # loss function - cross entropy loss
    # optimizer - stochastic gradient descent with learning rate lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(d_network.parameters(), lr=learning_rate)

    # training phase for epochs number of iterations
    # feed forward inputs with the available weights and bias to produce output
    # loss is calculated by cross entropy loss function for original class value and output from feed forward
    # calculating gradient adjustments for the weights and bias backward
    # updating weights and bias using SGD
    print("\nTraining starts here")
    for i in range(epochs):
        optimizer.zero_grad()
        output = d_network.forward(X_train)
        loss = criterion(output, y_train)
        print("Loss on Epoch {} ====> {}".format(i, loss.item()))
        loss.backward()
        optimizer.step()

    _, predict_y = torch.max(d_network(X_train), 1)
    print("\nTraining Accuracy: ", accuracy_score(y_train, predict_y.data))

    _, predict_y = torch.max(d_network(X_test), 1)
    print("\nTesting Accuracy: ", accuracy_score(y_test, predict_y.data))

    # predicting diabetes with an instance from the testing data set

    test = 17
    x_input = X_test[[test]]
    output = d_network(x_input)
    predicted = torch.argmax(output)

    print("\nChecking for a single input")
    print("\nInstance tested")
    print(x_input)
    print("\nOutput probability from network")
    print(output)
    print("\nPredicted class")
    print(predicted.data)
    print("\nOriginal class")
    print(y_test.iloc[test])