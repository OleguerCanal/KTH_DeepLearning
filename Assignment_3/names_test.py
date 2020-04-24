
import numpy as np
import sys, pathlib
from helper import read_names
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, _, _ = read_names(n_train=-1)

    # Compute class count for normalization
    class_sum = np.sum(y_train, axis=1)*y_train.shape[0]
    class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)

    print(x_train.shape)
    print(np.average(y_train, axis=1))
    print(class_sum)
    print(class_count)
    
    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define hyperparams
    d = x_train.shape[0]
    n1 = 20  # Filters of first Conv2D
    k1 = 5  # First kernel
    n2 = 20  # Filters of second Conv2D
    k2 = 3

    # Define model
    model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    model.add(Conv2D(num_filters=n1, kernel_shape=(d, k1), input_shape=x_train.shape[:-1]))
    model.add(Relu())
    model.add(Conv2D(num_filters=n2, kernel_shape=(1, k2)))
    model.add(Relu())
    model.add(Flatten())
    model.add(Dense(nodes=y_train.shape[0]))
    model.add(Softmax())

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=1000, lr = 1e-2, momentum=0.7, l2_reg=0.001,
              compensate=False, callbacks=callbacks)
    model.save("models/names_no_compensation")

    mt.plot_training_progress()
    # y_pred_prob = model.predict(x_train)