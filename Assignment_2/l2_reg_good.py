import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.layers import Dense, Softmax, Relu, Dropout
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import Accuracy
from mlp.utils import LoadXY
import matplotlib.pyplot as plt

np.random.seed(1)


if __name__ == "__main__":
    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    for i in [2, 3, 4, 5]:
        x, y = LoadXY("data_batch_" + str(i))
        x_train = np.concatenate((x_train, x), axis=1)
        y_train = np.concatenate((y_train, y), axis=1)
    x_val = x_train[:, -1000:]
    y_val = y_train[:, -1000:]
    x_train = x_train[:, :-1000]
    y_train = y_train[:, :-1000]
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Dense(nodes=800, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=800))
    model.add(Softmax())

    ns = 800

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  # Modifies lr while training
    callbacks = [mt, lrs]

    # Fit model
    iterations = 6*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, iterations=iterations,
            l2_reg=10**-1.85, shuffle_minibatch=True,
            callbacks=callbacks)
    model.save("models/l2reg_optimization_good")
    
    # Test model
    val_acc = model.get_metric_loss(x_val, y_val)[0]
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "Test acc: " + str(test_acc)
    mt.plot_training_progress(show=True, save=True, name="figures/l2reg_optimization/good", subtitle=subtitle)
    print("Val accuracy:", val_acc)
    print("Test accuracy:", test_acc)
