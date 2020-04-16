
import numpy as np
import sys, pathlib
from helper import read_mnist
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_mnist(n_train=10, n_test=2)

    # print(x_train.shape)
    # print(x_train[:, :, 0, 0])
    # for i in range(x_train.shape[-1]):
    #     cv2.imshow("example", x_train[:, :, 0, i])
    #     cv2.waitKey(0)

    x_train = np.zeros((5, 5, 1, 1))
    x_train[0, 0, 0, 0] = 2
    x_train[1, 3, 0, 0] = 1
    x_train[4, 0, 0, 0] = 3
    print(x_train)


    conv = Conv2D(num_filters = 2, kernel_shape = (3, 3, 1))
    outputs = conv(x_train)
    print(outputs.shape)

    # print(np.max(outputs))
    print(outputs)

    # for i in range(x_train.shape[-1]):
    #     cv2.imshow("example", outputs[:, :, 0, i]/np.max(outputs))
    #     cv2.waitKey(0)
