import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.layers import Dense, Softmax, Relu, Dropout
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import Accuracy
from mlp.utils import LoadXY

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
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
    model.add(Dropout(ones_ratio=0.50))
    model.add(Dense(nodes=10, input_dim=800))
    model.add(Softmax())

    ns = 500

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    # bms = BestModelSaver(save_dir=None)  # Saves model with highest val_metric
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-3, lr_max=1e-1, ns=ns)  # Modifies lr while training
    # callbacks = [mt, bms, lrs]
    callbacks = [mt, lrs]

    # Fit model
    iterations = 4*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, iterations=iterations, momentum=0.89,
            l2_reg=1e-5, shuffle_minibatch=True,
            callbacks=callbacks)
    model.save("models/yes_dropout_test")
    
    # Test model
    # best_model = bms.get_best_model()
    # test_acc, test_loss = best_model.get_metric_loss(x_test, y_test)
    # subtitle = "No Dropout, Test acc: " + test_acc
    subtitle = ""
    mt.plot_training_progress(show=True, save=True, name="figures/test_dropout_test", subtitle=subtitle)
    # print("Test accuracy:", test_acc)