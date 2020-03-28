# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time

from mlp.models import Sequential
from mlp.layers import Activation, Dense
from mlp.utils import getXY, LoadBatch, prob_to_class

def evaluate_cost(W, x, y_real):
    model.layers[0].weights = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real)
    return c

def ComputeGradsNum(x, y_real, model, h):
    """ Converted from matlab code """
    print("COmputing numerical gradients...")
    W = copy.deepcopy(model.layers[0].weights)

    no 	= 	W.shape[0]
    d 	= 	x.shape[0]

    # c = evaluate_cost(W, x, y_real)
    grad_W = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.matrix(W)
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real)
            
            W_try = np.matrix(W)
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W

if __name__ == "__main__":
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    x_val, y_val = getXY(LoadBatch("data_batch_2"))
    x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0:20]
    y = y_train[:, 0:20]
    reg = 0.1

    # Define model
    model = Sequential(loss="cross_entropy", reg_term=reg)
    model.add(Dense(nodes=10, input_dim=x.shape[0], weight_initialization="fixed"))
    model.add(Activation("softmax"))

    anal_time = time.time()
    analytical_grad = model.fit(x, y,
                                batch_size=10000, epochs=1, lr=0, # 0 lr will not change weights
                                momentum=0, l2_reg=reg)
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, h=0.001)
    num_time = num_time - time.time()

    _EPS = 0.0000001
    denom = np.abs(analytical_grad) + np.abs(numerical_grad)
    error = np.average(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    print("Averaged Element-Wise Relative Error:", error*100, "%")
    print("Speedup:", (num_time/anal_time))

