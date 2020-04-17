
import numpy as np
import sys, pathlib
from helper import read_mnist
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

if __name__ == "__main__":
    # x_train, y_train, x_test, y_test = read_mnist(n_train=10, n_test=2)

    # # print(x_train.shape)
    # # print(x_train[:, :, 0, 0])
    # # for i in range(x_train.shape[-1]):
    # #     cv2.imshow("example", x_train[:, :, 0, i])
    # #     cv2.waitKey(0)

    # x_train = np.zeros((5, 5, 1, 1))
    # x_train[0, 0, 0, 0] = 2
    # x_train[1, 3, 0, 0] = 1
    # x_train[4, 0, 0, 0] = 3
    # # print(x_train)

    # conv = Conv2D(num_filters = 2, kernel_shape = (3, 3, 1))
    # outputs = conv(x_train)
    # print(outputs.shape)
    # print(outputs)

    # # print(np.max(outputs))
    # # print(outputs[1, 1, 0, 0])

    # # for i in range(x_train.shape[-1]):
    # #     cv2.imshow("example", outputs[:, :, 0, i]/np.max(outputs))
    # #     cv2.waitKey(0)

    # x_train, y_train, x_test, y_test = read_mnist(n_train=2, n_test=2)
    
    # model = Sequential(loss=CrossEntropy())
    # model.add(Conv2D(num_filters = 2, kernel_shape = (3, 3, 1)))
    # model.add(Flatten())
    # model.add(Dense(nodes=10, input_dim=))
    # model.add(Softmax())
    # y_pred_prob = model.predict(x_train)
    # print(y_pred_prob.shape)
    # print(y_pred_prob)

    layers = []
    conv1 = Conv2D(num_filters=5, kernel_shape=(3, 3), input_shape=(28, 28, 1))
    print(conv1.input_shape)
    print(conv1.output_shape)
    print(conv1.filters.shape)
    conv2 = Conv2D(num_filters=10, kernel_shape=(6, 7))
    conv2.compile(input_shape=conv1.output_shape)
    print(conv2.input_shape)
    print(conv2.output_shape)
    print(conv2.filters.shape)
    flat = Flatten()
    flat.compile(input_shape=conv2.output_shape)
    print(flat.input_shape)
    print(flat.output_shape)
    dense1 = Dense(nodes=10)
    dense1.compile(input_shape=flat.output_shape)
    print(dense1.input_shape)
    print(dense1.output_shape)
    soft = Softmax()
    soft.compile(dense1.output_shape)
    print(soft.input_shape)
    print(soft.output_shape)

    # dense1 = Dense(nodes=3, input_dim=5)
    # sf1 = Softmax()
    # drp = Dropout(ones_ratio=0.5)
    # # sf1.compile(input_shape=dense1.output_shape)
    # drp.compile(input_shape=dense1.output_shape)
    # print(drp.input_shape)
    # print(drp.output_shape)
    # # dense2 = Dense(nodes=10)
    # # dense2.compile(dense1.output_shape)
    # print(dense1.output_shape)
    # # print(sf1.input_shape)
    # # print(sf1.output_shape)
    # # print(dense2.is_compiled)
    # # print(dense2.is_compiled)