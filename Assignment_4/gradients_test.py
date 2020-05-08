import numpy as np
from helper import load_data
import copy
import sys, pathlib
from tqdm import tqdm
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import VanillaRNN
from mlp.batchers import RnnBatcher
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import one_hotify

np.random.seed(0)
np.random.seed = 0


def evaluate_cost(W, x, y_real, l2_reg):
    model.layers[0].U = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")
    W = copy.deepcopy(model.layers[0].U)

    no 	= 	W.shape[0]
    d 	= 	x.shape[0]

    # c = evaluate_cost(W, x, y_real)
    grad_W = np.zeros(W.shape)
    for i in tqdm(range(W.shape[0])):
        for j in range(W.shape[1]):
            W_try = np.array(copy.deepcopy(W))
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            W_try = np.array(copy.deepcopy(W))
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W


def print_error(analytical_grad_weight, numerical_grad_w):
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


if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/test.txt")

    # Define callbacks
    # mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    # callbacks = [mt]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 3  # n

    batcher = RnnBatcher(seq_length)

    # Define model
    v_rnn = VanillaRNN(state_size=5, input_size=K, output_size=K)
    model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    model.add(v_rnn)

    state = np.array(np.random.normal(0.1, 1./100.,
            (v_rnn.state_size,1)))

    v_rnn.reset_state(state)

    # Fit model
    l2_reg = 0.0
    model.fit(X=encoded_data, epochs=1, lr = 2e-2, momentum=0.95, l2_reg=l2_reg,
              batcher=batcher, callbacks=[])
    # print(model.layers[0].dl_dw)
    print(model.layers[0].dl_du)
    # print(model.layers[0].dl_dv)
    anal = copy.deepcopy(model.layers[0].dl_du)

    model.layers[0].reset_state(state)

    x = np.array((char_to_ind['o'], char_to_ind['l'], char_to_ind['i'], ))
    x = one_hotify(x, num_classes = K)

    y = np.array((char_to_ind['l'], char_to_ind['i'], char_to_ind['s'], ))
    y = one_hotify(y, num_classes = K)

    grad_w = ComputeGradsNum(x, y, model, l2_reg, h=1e-8)
    num = copy.deepcopy(grad_w)
    print("num")
    print(grad_w)

    print_error(anal, num)
    # pred = [ind_to_char[elem] for elem in np.argmax(x, axis=0)]
    # print(pred)
    # print(x)
    # print(x.shape)
    # for i in range(5):
    #     probs = v_rnn(x)
    #     next_elem = np.argmax(probs, axis=0)
    #     pred = [ind_to_char[elem] for elem in next_elem]
    #     print(pred)
    #     x = one_hotify(next_elem, K)

    # mt.plot_training_progress(save=True, name="figures/names_best")