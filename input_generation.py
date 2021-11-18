#When using this sample file, please make sure that the sample rate is 0.01s.
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt,butter
from sklearn.decomposition import PCA

def visualization(X,Y,Z):
    plt.plot(X, label='X')
    plt.plot(Y, label='Y')
    plt.plot(Z, label='Z')
    plt.legend()
    plt.show()

def readxzya(data,Time):

    time_interval = (Time[20] - Time[10]) / 10

    if time_interval < 0.008:
        frequency = int(np.trunc(0.1 / time_interval))
        X = np.array(data['Acceleration x (m/s^2)'])
        X = X[::frequency]
        Y = np.array(data['Acceleration y (m/s^2)'])
        Y = Y[::frequency]
        Z = np.array(data['Acceleration z (m/s^2)'])
        Z = Z[::frequency]

    else:
        frequency = int(np.trunc(0.1 / time_interval))
        X = np.array(data['X (m/s^2)'])
        X = X[::frequency]
        Y = np.array(data['Y (m/s^2)'])
        Y = Y[::frequency]
        Z = np.array(data['Z (m/s^2)'])
        Z = Z[::frequency]

    return X,Y,Z


def readxzyg(data, Time):

    time_interval = (Time[20] - Time[10]) / 10

    if time_interval < 0.008:

        frequency = int(np.round(0.1/time_interval))
        X = np.array(data['Gyroscope x (rad/s)'])
        X = X[::frequency]
        Y = np.array(data['Gyroscope y (rad/s)'])
        Y = Y[::frequency]
        Z = np.array(data['Gyroscope z (rad/s)'])
        Z = Z[::frequency]

    else:
        frequency = int(np.round(0.1/time_interval))
        X = np.array(data['X (rad/s)'])
        X = X[::frequency]
        Y = np.array(data['Y (rad/s)'])
        Y = Y[::frequency]
        Z = np.array(data['Z (rad/s)'])
        Z = Z[::frequency]

    return X, Y, Z

def dataprocess(X,Y,Z):

    # Use the butter filter
    sos = butter(8, 0.125, output='sos')
    X_filtered = sosfiltfilt(sos, X)
    Y_filtered = sosfiltfilt(sos, Y)
    Z_filtered = sosfiltfilt(sos, Z)

    # find the starting point
    id_min = 0
    X_max = np.max(X_filtered)
    for idx in range(len(X_filtered)):
        if X_filtered[idx] > 3 * X_max / 4:
            id_min = idx
            break
    # sample the data of 10 seconds
    length = 100
    X_sampled = X_filtered[id_min:id_min + length]
    Y_sampled = Y_filtered[id_min:id_min + length]
    Z_sampled = Z_filtered[id_min:id_min + length]
    # Do the PCA
    data = np.stack([X_sampled, Y_sampled, Z_sampled], axis=1)
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)
    x_pca = data_pca[:, 0]
    y_pca = data_pca[:, 1]
    z_pca = data_pca[:, 2]
    #visualization(x_pca,y_pca,z_pca)

    return x_pca,y_pca,z_pca















