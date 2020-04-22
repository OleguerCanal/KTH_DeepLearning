
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
from mlp.utils import LoadXY

def get_data(n_train=None, n_val=None, n_test=None):
    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    if n_train is not None:
        x_train = x_train[..., 0:n_train]
        y_train = y_train[..., 0:n_train]
    if n_val is not None:
        x_val = x_val[..., 0:n_val]
        y_val = y_val[..., 0:n_val]
    if n_test is not None:
        x_test = x_test[..., 0:n_test]
        y_test = y_test[..., 0:n_test]

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # reshaped_train = np.zeros((32, 32, 3, x_train.shape[-1]))
    # for i in range(x_train.shape[-1]):
    #     flatted_image = np.array(x_train[..., i])
    #     image = np.reshape(flatted_image,  (32, 32, 3), order='F')
    #     cv2.imshow("image", image)
    #     cv2.waitKey()

    x_train = np.reshape(np.array(x_train), (32, 32, 3, x_train.shape[-1]), order='F')
    x_val = np.reshape(np.array(x_val), (32, 32, 3, x_val.shape[-1]), order='F')
    x_test = np.reshape(np.array(x_test), (32, 32, 3, x_test.shape[-1]), order='F')

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    # Load data
    # x_train, y_train, x_val, y_val, x_test, y_test = get_data(n_train=200, n_val=200, n_test=2)
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(n_train=6000, n_val=1000, n_test=200)

    print(x_train.shape)
    # print(y_train.shape)

    # for i in range(200):
    #     cv2.imshow("image", x_train[..., i])
    #     cv2.waitKey()

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    bms = BestModelSaver("models/best_cifar")  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=64, kernel_shape=(6, 6), input_shape=(32, 32, 3)))
    model.add(Relu())
    model.add(Conv2D(num_filters=32, kernel_shape=(3, 3)))
    model.add(Relu())
    model.add(Flatten())
    # model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(nodes=400))
    model.add(Relu())
    model.add(Dense(nodes=10))
    model.add(Softmax())

    # for filt in model.layers[0].filters:
    #     print(filt)
    # y_pred_prob = model.predict(x_train)
    # print(y_pred_prob)

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=200, epochs=50, lr = 1e-2, momentum=0.5, callbacks=callbacks)
    model.save("models/cifar_test")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))
