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

def evaluator(l2_reg):
    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Dense(nodes=800, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=800))
    model.add(Softmax())

    ns = 800

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-3, lr_max=1e-1, ns=ns)  # Modifies lr while training
    callbacks = [mt, lrs]

    # Fit model
    iterations = 4*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, iterations=iterations,
            l2_reg=l2_reg, shuffle_minibatch=True,
            callbacks=callbacks)
    model.save("models/yes_dropout_test")
    
    # Test model
    val_acc = model.get_metric_loss(x_val, y_val)[0]
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "L2 param: " + str(l2_reg) + ", Test acc: " + str(test_acc)
    mt.plot_training_progress(show=True, save=True, name="figures/l2reg_optimization/" + str(l2_reg), subtitle=subtitle)
    print("Val accuracy:", val_acc)
    print("Test accuracy:", test_acc)
    return val_acc


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

    # Coarse search
    l2_params = np.linspace(-7.0, -1.0, num=80)
    # val_accuracies = []
    # for l2_param in l2_params:
    #     val_acc = evaluator(10**l2_param)
    #     val_accuracies.append(val_acc)
    #     np.save("l2_search_vals", val_accuracies)
    val_accuracies = np.load("l2_search_vals.npy")
    ratio = 10
    minimum = 50
    maximum = 75
    l2_params = [l2_params[i] for i in range(80) if i > minimum and i < maximum]
    val_accuracies = [val_accuracies[i] for i in range(80) if i > minimum and i < maximum]

    plt.plot(l2_params, val_accuracies)
    plt.xlabel("l2 regularization")
    plt.ylabel("Validation Accuracy")
    plt.ylim(bottom=0.4, top=0.5)
    plt.title("Uniform Search Results")
    plt.savefig("figures/l2reg_search_results_2.png")
    plt.show()
