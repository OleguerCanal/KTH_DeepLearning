import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/HyperParameter-Optimizer/")

from decimal import Decimal
from skopt.space import Real, Integer, Categorical
from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Dense, Softmax, Relu
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from util.misc import dict_to_string

from gaussian_process import GaussianProcessSearch

def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", **kwargs):
    print(kwargs)
    # Saving directories
    figure_file = "figures/" + experiment_name + "/" + dict_to_string(kwargs)
    model_file = "models/" + experiment_name + "/" + dict_to_string(kwargs)

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Dense(nodes=kwargs["hidden_units"], input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=kwargs["hidden_units"]))
    model.add(Softmax())

    # Pick metaparams
    batch_size = 100
    ns = kwargs["ns"]
    iterations = kwargs["number_of_cycles"]*ns

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    bms = BestModelSaver(save_dir=None)
    lrs = LearningRateScheduler(
        evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  
    callbacks = [mt, bms, lrs]
    # callbacks = [mt, lrs]

    # Adjust logarithmic
    kwargs["l2_reg"] = 10**kwargs["l2_reg"]

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=batch_size, epochs=None, iterations=iterations,
              callbacks=callbacks, **kwargs)

    # Write results    
    best_model = bms.get_best_model()
    test_acc = best_model.get_metric_loss(x_test, y_test)[0]
    
    l2_str = str("{:.2E}".format(Decimal(kwargs["l2_reg"])))
    ns_str = str(ns)
    nc_str = str(kwargs["number_of_cycles"])
    h_units_str = str(kwargs["hidden_units"])
    m_str = str("{:.2E}".format(Decimal(kwargs["momentum"])))
    test_acc_str = str("{:.2E}".format(Decimal(test_acc)))

    subtitle = "l2reg:" + l2_str + ", ns:" + ns_str + ", nc:" + nc_str +\
        ", units:" + h_units_str + ", moment:" + m_str + ", Test Acc: " + test_acc_str
    mt.plot_training_progress(show=False, save=True, name=figure_file, subtitle=subtitle)

    # Maximizing value: validation accuracy
    val_metric = bms.best_metric
    # val_metric = model.get_metric_loss(x_val, y_val)[0]
    return val_metric


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

    fixed_args = {
        "experiment_name": "metaparam_search",
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "shuffle_minibatch":True,
    }

    pathlib.Path(fixed_args["experiment_name"]).mkdir(parents=True, exist_ok=True)
    
    # Search space
    l2reg_space = Real(name='l2_reg', low=-5, high=-1)
    cycles_length = Integer(name='ns', low=400, high=1200)
    number_of_cycles = Integer(name="number_of_cycles", low=2, high=6)
    hidden_units = Integer(name="hidden_units", low=50, high=200)
    momentum = Real(name="momentum", low=0.2, high=0.95)
    search_space = [l2reg_space, cycles_length, number_of_cycles, hidden_units, momentum]

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_args,
                                      evaluator=evaluator,
                                      input_file=None,
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls=80,
                                 n_random_starts=20,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)
