import numpy as np
import sys, pathlib
from helper import load_data
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import VanillaRNN, Softmax
from mlp.batchers import RnnBatcher
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler, TextSynthesiser
from mlp.utils import one_hotify, generate_sequence

np.random.seed(0)

experiment_name = "lstm_keras"

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def get_dataset(X, seq_length = 25):
    batcher = RnnBatcher(seq_length)
    x_train = []
    y_train = []
    for X_minibatch, Y_minibatch in batcher(X):
        x_train.append(X_minibatch)
        y_train.append(Y_minibatch)
    return np.array(x_train), np.array(y_train)
        
if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/goblet_book.txt")

    seq_length = 25
    x_train, y_train = get_dataset(encoded_data, seq_length)

    print(x_train.shape)

    # model = Sequential()
    # model.add(LSTM(100, input_shape=(seq_length, )))
    # model.add(Dense(seq_length))
    # model.compile(loss="categorical_crossentropy", optimizer='adam')
    # model.fit(x_train, y_train, epochs=2, batch_size=50, verbose=2)