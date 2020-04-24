
import numpy as np
import sys, pathlib
from helper import read_mnist, read_cifar_10, read_names
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

np.random.seed(0)

def evaluate_cost_W(W, x, y_real, l2_reg, filter_id):
    model.layers[0].filters[filter_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def evaluate_cost_b(W, x, y_real, l2_reg, bias_id):
    model.layers[0].biases[bias_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")

    grads_w = []
    for filter_id, filt in enumerate(model.layers[0].filters):
        W = copy.deepcopy(filt)  # Compute W
        grad_W = np.zeros(W.shape)
        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                for c in range(W.shape[2]):
                    W_try = np.array(W)
                    # print(W_try.shape)
                    W_try[i,j,c] -= h
                    c1 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    W_try = np.array(W)
                    W_try[i,j,c] += h
                    c2 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    grad_W[i,j,c] = (c2-c1) / (2*h)

                    model.layers[0].filters[filter_id] = W  # Reset it

        grads_w.append(grad_W)

    grads_b = []
    for bias_id, bias in enumerate(model.layers[0].biases):
        b_try = copy.deepcopy(bias) - h
        c1 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        b_try = copy.deepcopy(bias) + h
        c2 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        grad_b = (c2-c1) / (2*h)
        model.layers[0].biases[bias_id] = bias  # Reset it

        grads_b.append(grad_b)
    return grads_w, grads_b

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=3, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=2, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_names(n_train=500)

    class_sum = np.sum(y_train, axis=1)*y_train.shape[0]
    class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)

    print(class_count)

    print(type(x_train[0, 0, 0]))

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=2, kernel_shape=(4, 4), input_shape=x_train.shape[0:-1]))
    model.add(Relu())
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(nodes=y_train.shape[0]))
    model.add(Relu())
    model.add(Softmax())

    reg = 0.0

    # Fit model
    anal_time = time.time()
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=200, epochs=1, lr=0, momentum=0, l2_reg=reg)
    analytical_grad_weight = model.layers[0].filter_gradients
    analytical_grad_bias = model.layers[0].bias_gradients
    # print(analytical_grad_weight)
    print(analytical_grad_bias)
    anal_time = time.time() - anal_time

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad_w, numerical_grad_b = ComputeGradsNum(x_train, y_train, model, l2_reg=reg, h=1e-7)
    # print(numerical_grad_w)
    print(numerical_grad_b)
    num_time = time.time() - num_time

    print("Weight Error:")
    _EPS = 0.0000001
    denom = np.abs(analytical_grad_weight) + np.abs(numerical_grad_w)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")


    print("Bias Error:")
    _EPS = 0.0000001
    denom = np.abs(analytical_grad_bias) + np.abs(numerical_grad_b)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")

    print("Speedup:", (num_time/anal_time))

