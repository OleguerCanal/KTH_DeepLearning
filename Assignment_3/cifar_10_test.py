
import numpy as np
import sys, pathlib
from helper import read_mnist
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import LoadXY

from helper import read_cifar_10

if __name__ == "__main__":
    # Load data
    # x_train, y_train, x_val, y_val, x_test, y_test = get_data(n_train=200, n_val=200, n_test=2)
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=15000, n_val=200, n_test=200)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10()

    print(x_train.shape)
    # print(y_train.shape)

    # for i in range(200):
    #     cv2.imshow("image", x_train[..., i])
    #     cv2.waitKey()

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    bms = BestModelSaver("models/best_cifar")  # Stores training evolution info (losses and metrics)
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-3, lr_max=0.2, ns=500)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt, lrs]

    # Define architecture (copied from https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/)
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=32, kernel_shape=(3, 3), stride=2, input_shape=(32, 32, 3)))
    model.add(Relu())
    model.add(Conv2D(num_filters=64, kernel_shape=(3, 3)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2), stride=2))
    model.add(Conv2D(num_filters=128, kernel_shape=(2, 2)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2)))
    model.add(Flatten())
    model.add(Dense(nodes=200))
    model.add(Relu())
    model.add(Dense(nodes=10))
    model.add(Softmax())


    # for filt in model.layers[0].filters:
    #     print(filt)
    # y_pred_prob = model.predict(x_train)
    # print(y_pred_prob)

    # Fit model
    model.load("models/cifar_test_2")
    mt.load("models/tracker")
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=20, momentum=0.9, l2_reg=0.003, callbacks=callbacks)
    model.save("models/cifar_test_2")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    # y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))
