import numpy as np
import sys
import pathlib
from helper import read_names

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/HyperParameter-Optimizer/")

from decimal import Decimal
from skopt.space import Real, Integer, Categorical
from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from util.misc import dict_to_string

from gaussian_process import GaussianProcessSearch

def evaluator(x_train, y_train, x_val, y_val, experiment_name="", **kwargs):
    print(kwargs)
    # Saving directories
    figure_file = "figures/" + experiment_name + "/" + dict_to_string(kwargs)
    model_file = "models/" + experiment_name + "/" + dict_to_string(kwargs)

    mt = MetricTracker()  # Stores training evolution info (losses and metrics)

    # Define model
    d = x_train.shape[0]
    n1 = kwargs["n1"]  # Filters of first Conv2D
    k1 = kwargs["k1"]  # First kernel y size
    n2 = kwargs["n2"]  # Filters of second Conv2D
    k2 = kwargs["k2"]  # Second kernel y size
    batch_size = kwargs["batch_size"]

    try:
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
                batch_size=batch_size, epochs=1000, lr = 1e-2, momentum=0.8, l2_reg=0.001,
                compensate=True, callbacks=[mt])
    except Exception as e:
        print(e)
        return -1  # If configuration impossible
    model.save(model_file)

    # Write results    
    n1 = str(n1)
    n2 = str(n2)
    k1 = str(k1)
    k2 = str(k2)
    batch_size = str(batch_size)
    subtitle = "n1:" + n1 + ", n2:" + n2 + ", k1:" + k1 + ", k2:" + k1 +\
               ", batch_size:" + batch_size
    mt.plot_training_progress(show=False, save=True, name=figure_file, subtitle=subtitle)

    # Maximizing value: validation accuracy
    return model.val_metric


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, _, _ = read_names(n_train=-1)

    fixed_args = {
        "experiment_name": "name_metaparam_search",
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }
    pathlib.Path(fixed_args["experiment_name"]).mkdir(parents=True, exist_ok=True)
    
    # Search space
    n1 = Integer(name='n1', low=10, high=40)
    n2 = Integer(name='n2', low=10, high=40)
    k1 = Integer(name='k1', low=2, high=10)
    k2 = Integer(name='k2', low=2, high=10)
    batch_size = Integer(name="batch_size", low=50, high=300)
    search_space = [n1, n2, k1, k2, batch_size]

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_args,
                                      evaluator=evaluator,
                                      input_file=None,
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls = 8,
                                 n_random_starts=5,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)
