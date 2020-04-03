# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY, prob_to_class
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string


def evaluate_cost(W, x, y_real, l2_reg):
    model.layers[0].weights = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")
    W = copy.deepcopy(model.layers[0].weights)

    no 	= 	W.shape[0]
    d 	= 	x.shape[0]

    # c = evaluate_cost(W, x, y_real)
    grad_W = np.zeros(W.shape)
    for i in tqdm(range(W.shape[0])):
        for j in range(W.shape[1]):
            W_try = np.matrix(W)
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            W_try = np.matrix(W)
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W

if __name__ == "__main__":
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0]
    y = y_train[:, 0]
    reg = 0.1

    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    anal_time = time.time()
    model.fit(x, y, batch_size=None, epochs=1, lr=0, # 0 lr will not change weights
                    momentum=0, l2_reg=reg)
    analytical_grad = model.layers[0].gradient
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, l2_reg=reg, h=0.001)
    num_time = num_time - time.time()

    _EPS = 0.0000001
    denom = np.abs(analytical_grad) + np.abs(numerical_grad)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")
    print("Speedup:", (num_time/anal_time))

