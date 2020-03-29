import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.utils import getXY, LoadBatch, prob_to_class
from mlp.layers import Activation, Dense
from mlp.models import Sequential
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    for i in [2, 3, 4, 5]:
        x, y = getXY(LoadBatch("data_batch_" + str(i)))
        x_train = np.concatenate((x_train, x), axis=1)
        y_train = np.concatenate((y_train, y), axis=1)
    x_val = x_train[:, -1000:]
    y_val = y_train[:, -1000:]
    x_train = x_train[:, :-1000]
    y_train = y_train[:, :-1000]
    x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Modelling
    model = Sequential(loss="categorical_hinge")
    model.add(Dense(nodes=10, input_dim=x.shape[0], weight_initialization="fixed"))

    best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=20, epochs=100, lr=0.001, # 0 lr will not change weights
                        momentum=0.5, l2_reg=0.05, save_path="models/svm/test_1")
    best_model.plot_training_progress(show=False,
                                save=True,
                                name="figures/svm/test_1",
                                subtitle="subtitle")
    test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    val_acc = best_model.get_classification_metrics(x_val, y_val)[0]
    print("test_acc:", test_acc)
    print("val_acc:", val_acc)