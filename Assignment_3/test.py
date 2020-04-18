
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
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=1000, n_val=5, n_test=2)

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=10, kernel_shape=(5, 5), input_shape=(28, 28, 1)))
    model.add(Relu())
    model.add(Conv2D(num_filters=20, kernel_shape=(3, 3)))
    model.add(Relu())
    model.add(Flatten())
    model.add(Dense(nodes=128))
    model.add(Relu())
    model.add(Dense(nodes=10))
    model.add(Softmax())

    # for filt in model.layers[0].filters:
    #     print(filt)
    # y_pred_prob = model.predict(x_train)
    # print(y_pred_prob)

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=10, lr = 1e-2, momentum=0.5, callbacks=callbacks)

    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    # y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(y_pred_prob)
