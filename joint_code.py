#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/test_gp_search.py contains:
 #####################################################
import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch

from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import json


a_space = Real(name='lr', low=0., high=1.)
b_space = Integer(name='batch_size', low=0, high=200)
c_space = Real(name='alpha', low=0, high=1)
d_space = Real(name='some_param', low=0, high=100)

search_space = [a_space, b_space, c_space, d_space]
fixed_space = {'noise_level': 0.1}


def func(lr, batch_size, alpha, some_param, noise_level):
    # Max = 101
    return lr**3 + batch_size**2 + some_param * alpha + np.random.randn() * noise_level


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func,
                                  input_file=None,  # Use None to start from zero
                                  output_file='test.csv')
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=10, n_random_starts=0,
                             noise=fixed_space['noise_level'],
                             verbose=True,
                             )

x = gp_search.get_next_candidate(n_points=5)
print('NEXT CANDIDATES: ' + str(x))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/load_save.py contains:
 #####################################################
import os
import json
import pandas


def load(data_file):
    """Load the given data file and returns a dictionary of the values

    Args:
        data_file (str): Path to a file containing data in one of the known formats (json, csv)

    Returns:
        A dictionary with the loaded data

    """
    _, ext = os.path.splitext(data_file)
    try:
        return known_file_types[ext]['load'](data_file)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def save(data_file, dictionary):
    """Save the given dictionary in the given file. Format is determined by data_file extension

    Args:
        data_file (str): Path to a file in which to save the data. Extension is used to determine
        the format, therefore the path must contain an extension.
        dictionary (dict): Dictionary with the data

    """
    _, ext = os.path.splitext(data_file)
    try:
        known_file_types[ext]['save'](data_file, dictionary)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def _load_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)


def _save_json(data_file, dictionary):
    with open(data_file, 'w') as file:
        json.dump(dictionary, file)


def _load_csv(data_file):
    data_frame = pandas.read_csv(data_file, header=0)
    res = {}
    for k in data_frame:
        res[k] = data_frame[k].tolist()
    return res


def _save_csv(data_file, dictionary):
    data_frame = pandas.DataFrame.from_dict(data=dictionary, orient='columns')
    data_frame.to_csv(data_file, header=True, index=False)


known_file_types = {
    '.json':
        {
            'load': _load_json,
            'save': _save_json
        },
    '.csv':
        {
            'load': _load_csv,
            'save': _save_csv
        }
}



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/__init__.py contains:
 #####################################################


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/gaussian_process.py contains:
 #####################################################
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
from skopt import dump, load
import time
import matplotlib.pyplot as plt
import load_save
import sys
import pathlib
import numpy as np

# Session variables
session_params = {}


class GaussianProcessSearch:

    def __init__(self, search_space, fixed_space, evaluator, input_file=None, output_file=None):
        """Instantiate the GaussianProcessSearch and create the GaussianProcessRegressor

        Args:
            search_space (list): List of skopt.space.Dimension objects (Integer, Real,
                or Categorical) whose name must match the correspondent variable name in the
                evaluator function
            fixed_space (dict): Dictionary of parameters that will be passed by default to the
                evaluator function. The keys must match the correspondent names in the function.
            evaluator (function): Function of which we want to estimate the maximum. It must take
                the union of search_space and fixed_space as parameters and return a scalar value.
            input_file (str): Path to the file containing points in the search space and
                corresponding values that are already known.
            output_file (str): Path to the file where updated results will be stored.
        """
        self.search_space = search_space
        self.fixed_space = fixed_space
        self.evaluator = evaluator
        self.input_file = input_file
        self.output_file = output_file
        self.x_values = []
        self.y_values = []
        if input_file is not None:
            try:
                data_dict = load_save.load(data_file=input_file)
                self.x_values, self.y_values = self._extract_values(data_dict)
            except OSError as e:
                raise OSError('Cannot read input file. \n' + str(e))

    @staticmethod
    def _get_gp_regressor(length_scale=1., nu=2.5, noise=0.1):
        """Creates the GaussianProcessRegressor model

        Args:
            length_scale (Union[float, list]): Length scale of the GP kernel. If float, it is the
                same for all dimensions, if array each element defines the length scale of the
                dimension
            nu (float): Controls the smoothness of the approximation.
                see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        Returns:
            A skopt.learning.GaussianProcessRegressor with the given parameters

        """
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        return GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2)

    def get_maximum(self, n_calls=10, n_random_starts=5, noise=0.01, verbose=True,
                    plot_results=False):
        """Performs Bayesian optimization by iteratively evaluating the given function on points
        that are likely to be a global maximum.

        After the optimization, the evaluated values are stored in self.x_values and
        self.y_values and appended to the data file if provided.

        Args:
            n_calls (int): Number of iterations
            n_random_starts (int): Initial random evaluations if no previpus values are provided
            noise (float): Estimated noise in the data
            verbose (bool): Whether to print optimization details at each evaluation
            plot_results (bool): Whether to plot an analysis of the solution

        Returns:
            A tuple (x, y) with the argmax and max found of the evaluated function.

        """
        x_values = [x for x in self.x_values] if len(self.x_values) > 0 else None
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values] if len(self.y_values) > 0 else None
        rand_starts = 2 if len(self.x_values) == 0 and n_random_starts == 0 else n_random_starts
        res = gp_minimize(func=GaussianProcessSearch.evaluate,
                          dimensions=self.search_space,
                          n_calls=n_calls,
                          n_random_starts=rand_starts,
                          acq_func='EI',
                          acq_optimizer='lbfgs',
                          x0=x_values,
                          y0=y_values,
                          noise=noise,
                          n_jobs=-1,
                          callback=self.__save_res,
                          verbose=verbose)
        if plot_results:
            ax = plot_objective(res)
            plt.show()
            ax = plot_evaluations(res)
            plt.show()

        self.x_values = [[float(val) for val in point] for point in res.x_iters]
        self.y_values = [-val for val in res.func_vals]
        if self.output_file is not None:
            self.save_values()
            try:
                ax = plot_objective(res)
                plt.savefig( self.output_file + "_objective_plot.png")
            except Exception as e:
                print(e)
            try:
                ax = plot_evaluations(res)
                plt.savefig( self.output_file + "_evaluations_plot.png")
            except Exception as e:
                print(e)
        return res.x, -res.fun

    def add_point_value(self, point, value):
        """Add a point and the correspondent value to the knowledge.

        Args:
            point (Union[list, dict]): List of values correspondent to self.search_space
                dimensions (in the same order), or dictionary {dimension_name: value} for all
                the dimensions in self.search_space.
            value (float): Value of the function at the given point

        """
        p = []
        if isinstance(point, list):
            p = point
        elif isinstance(point, dict):
            for dim in self.search_space:
                p.append(point[dim.name])
        else:
            raise ValueError('Param point of add_point_value must be a list or a dictionary.')
        self.x_values.append(p)
        self.y_values.append(value)

    def get_next_candidate(self, n_points):
        """Returns the next candidates for the skopt acquisition function

        Args:
            n_points (int): Number of candidates desired

        Returns:
            List of points that would be chosen by gp_minimize as next candidate

        """
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values]
        optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator='gp',
            n_initial_points=len(self.x_values),
            acq_func='EI'
        )
        optimizer.tell(self.x_values, y_values)  # TODO Does this fit the values???
        points = optimizer.ask(n_points=n_points)
        return self._to_dict_list(points)

    def get_random_candidate(self, n_points):
        candidates = []
        for _ in range(n_points):
            candidate = {}
            for elem in self.search_space:
                candidate[str(elem.name)] = elem.rvs(n_samples=1)[0]
            candidates.append(candidate)
        return candidates

    def _to_dict_list(self, points):
        """Transform the list of points in a list of dictionaries {dimension_name: value}

        Args:
            points (list): List of lists of value, where for each list, the i-th element
            corresponds to a value for the i-th dimension of the search space

        Returns:
            A list of dictionaries, where each dictionary has the search space dimensions as keys
            and the correspondent value of points, in the self.search_space order

        """
        def to_dict(point):
            d = {}
            for i, dim in enumerate(self.search_space):
                d[dim.name] = point[i]
            return d
        return [to_dict(p) for p in points]

    def init_session(self):
        """Save in session variables. the parameters that will be passed to the evaluation function
        by default.

        """
        global session_params
        session_params['fixed_space'] = self.fixed_space
        session_params['evaluator'] = self.evaluator
        session_params['dimension_names'] = [dim.name for dim in self.search_space]

    def reset_session(self):
        """Reset session variables.

        """
        global session_params
        session_params = {}

    def _extract_values(self, data_dict):
        """Extracts the x values and target values from the given data dictionary.

         Args:
             data_dict (dict): A dictionaty like: {<param_name>: [list of values]} where all lists
                 have the same length and values at same index belong to the same point. The only
                 exception is data_dict['value'] that must contain a list of float correspondent
                 to the function evaluations in the points.

         Returns:
             A tuple (x_values, y_values) where
                 x_values (list): List of points in the search space
                 y_values (list): List of known values for the x_values points

        """
        y_values = data_dict['value']
        x_values = []
        for i, dimension in enumerate(self.search_space):
            name = dimension.name
            try:
                for j, v in enumerate(data_dict[name]):
                    if i == 0:  # If first dimension, instantiate an array for data point
                        x_values.append([])
                    x_values[j].append(data_dict[name][j])
            except KeyError:
                raise KeyError('Search space expects a ' + name + ' dimension but loaded data '
                                                                  'does not contain it')
        return x_values, y_values

    def _pack_values(self):
        """Packs the known values to a dictionary where keys are dimension names

        Returns: A dictionary {dimension_name: [dimension_values] for all dimensions,
            value: [result_values]}

        """
        res_dict = {}
        for i, dimension in enumerate(self.search_space):
            res_dict[dimension.name] = []
            for point in self.x_values:
                res_dict[dimension.name].append(point[i])
        res_dict['value'] = self.y_values
        return res_dict

    def save_values(self):
        """Save in the data file the known x_values and y_values

        """
        data_dict = self._pack_values()
        load_save.save(self.output_file, data_dict)

    @staticmethod
    def _to_key_value(values):
        """Transform the given list of values in a key-value dictionary from the search_space names

        Args:
            values (list): List of values of the same length as self.search_space

        Returns:
            A dictionary key[i]: value[i] where key[i] is the name of the i-th dimension of
            self.search_space and value[i] is the i-th element of values

        """
        global session_params
        name_value_dict = {}
        for i, name in enumerate(session_params['dimension_names']):
            name_value_dict[name] = values[i]
        return name_value_dict

    @staticmethod
    def evaluate(point):
        """Evaluate the evaluator function at the given point

        Args:
            point (list): List of values each one corresponding to a dimension of self.search_space

        Returns:
            The value of self.evaluator at the given point, negated (to be used in minimization)
        """
        global session_params
        evaluator_func = session_params['evaluator']
        fixed_space = session_params['fixed_space']
        # Transform the point in a mapping param_name=value
        name_value_dict = GaussianProcessSearch._to_key_value(point)
        args = {**fixed_space, **name_value_dict}
        return -evaluator_func(**args)

    def __save_res(self, res):
        t = time.time()
        pathlib.Path("gpro_results/").mkdir(parents=True, exist_ok=True)
        result_name = "gpro_results/" + str(t) + "_gpro_result.pkl" 
        dump(res, result_name)
        numpy_name = "gpro_results/" + str(t) + "_gpro_res.npy" 
        np.save(numpy_name, res.x)
        numpy_name = "gpro_results/" + str(t) + "_gpro_fun.npy" 
        np.save(numpy_name, -res.fun)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/metaparamoptimizer.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import itertools as it
import numpy as np
import pickle

# TODO(Oleguer): Think about the structure of all this

class MetaParamOptimizer:
    def __init__(self, save_path=""):
        self.save_path = save_path  # Where to save best result and remaining to explore
        pass

    def list_search(self, evaluator, dicts_list, fixed_args):
        """ Evaluates model (storing best) on provided list of param dictionaries
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        max_result = None
        for indx, evaluable_args in enumerate(dicts_list):
            print("MetaParamOptimizer evaluating:", indx, "/", len(dicts_list), ":", evaluable_args)
            args = {**evaluable_args, **fixed_args}  # Merge kwargs and evaluable_args dicts
            try:
                result = evaluator(**args)
            except Exception as e:
                print("MetaParamOptimizer: Exception found when evaluating:")
                print(e)
                print("Skipping to next point...")
                continue
            if (max_result is None) or (result["value"] > max_result["value"]):
                max_result = result
                max_result["best_params"] = evaluable_args
                self.save(max_result, name="metaparam_search_best_result")  # save best result found so far
            # Save remaning tests (in case something goes wrong, know where to keep testing)
            self.save(dicts_list[indx+1:], name="remaining_tests")
        return max_result

    def grid_search(self, evaluator, search_space, fixed_args):
        """ Performs grid search on specified search_space
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        points_to_evaluate = self.__get_all_dicts(search_space)
        return self.list_search(evaluator, points_to_evaluate, fixed_args)

    def GPR_optimizer(self, evaluator, search_space, fixed_args):
        pass # The other repo

    def save(self, elem, name="best_result"):
        """ Saves result to disk"""
        with open(self.save_path + "/" + name + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, name="best_model", path=None):
        if path is None:
            path = self.save_path
        with open(path + "/" + name, 'rb') as input:
            remaining_tests = pickle.load(input)
        return remaining_tests

    def __get_all_dicts(self, param_space):
        """ Given:
            dict of item: list(elems)
            returns:
            list (dicts of item : elem)
        """
        allparams = sorted(param_space)
        combinations = it.product(*(param_space[Name] for Name in allparams))
        dictionaries = []
        for combination in combinations:
            dictionary = {}
            for indx, name in enumerate(allparams):
                dictionary[name] = combination[indx]
            dictionaries.append(dictionary)
        return dictionaries

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/cyclical_lr_test.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.layers import Dense, Softmax, Relu
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
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    ns = 500

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    bms = BestModelSaver(save_dir=None)  # Saves model with highest val_metric
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  # Modifies lr while training
    callbacks = [mt, bms, lrs]

    # Fit model
    iterations = 2*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=100, epochs=None, iterations=iterations, lr=0.01, momentum=0.0,
                        l2_reg=0.01, shuffle_minibatch=False,
                        callbacks=callbacks)
    # model.save("models/mlp_overfit_test")
    mt.plot_training_progress(show=False, save=True, name="figures/mlp_cyclic_good")
    mt.plot_lr_evolution(show=False, save=True, name="figures/lr_cyclic_good")
    
    # Test model
    best_model = bms.get_best_model()
    test_acc, test_loss = best_model.get_metric_loss(x_test, y_test)
    print("Test accuracy:", test_acc)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/l2reg_search.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/cyclical_lr_good.py contains:
 #####################################################
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/HyperParameter-Optimizer/")

from skopt.space import Real
from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Dense, Softmax, Relu
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from util.misc import dict_to_string

from gaussian_process import GaussianProcessSearch

def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", **kwargs):
    # Saving directories
    figure_file = "figures/" + experiment_name + "/" + dict_to_string(kwargs)
    model_file = "models/" + experiment_name + "/" + dict_to_string(kwargs)

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    # Pick metaparams
    batch_size = 100
    ns = 2*np.floor(x_train.shape[1]/batch_size)
    iterations = 4*ns  # 2 cycles

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    # bms = BestModelSaver(save_dir=None)
    lrs = LearningRateScheduler(
        evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  
    # callbacks = [mt, bms, lrs]
    callbacks = [mt, lrs]

    # Adjust logarithmic
    kwargs["l2_reg"] = 10**kwargs["l2_reg"]

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=batch_size, epochs=None, iterations=iterations, **kwargs,
              callbacks=callbacks)

    # Write results    
    # best_model = bms.get_best_model()
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", Test Acc: " + str(test_acc)
    mt.plot_training_progress(show=False, save=True, name=figure_file, subtitle=subtitle)

    # Maximizing value: validation accuracy
    # val_metric = bms.best_metric
    val_metric = model.get_metric_loss(x_val, y_val)[0]
    return val_metric


if __name__ == "__main__":
    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    for i in [2, 3, 4, 5]:
        x, y = LoadXY("data_batch_" + str(i))
        x_train = np.concatenate((x_train, x), axis=1)
        y_train = np.concatenate((y_train, y), axis=1)
    x_val = x_train[:, -5000:]
    y_val = y_train[:, -5000:]
    x_train = x_train[:, :-5000]
    y_train = y_train[:, :-5000]
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    fixed_args = {
        "experiment_name": "l2reg_optimization",
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "momentum": 0.0,
        "shuffle_minibatch":False,
    }

    pathlib.Path(fixed_args["experiment_name"]).mkdir(parents=True, exist_ok=True)
    l2reg_space = Real(name='l2_reg', low=-7, high=-1)
    search_space = [l2reg_space]

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_args,
                                      evaluator=evaluator,
                                      input_file=None,
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls=15,
                                 n_random_starts=7,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/overfit_test.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY

np.random.seed(0)

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

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
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    # Fit model
    x_train = x_train[:, 0:100]
    y_train = y_train[:, 0:100]
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=100, epochs=200, lr=0.001, momentum=0.0,
                        l2_reg=0.0, shuffle_minibatch=False, save_path="models/mlp_overfit_test")
    model.plot_training_progress(save=True, name="figures/mlp_overfit_test")
    model.save("models/mlp_overfit_test")

    # Test model
    test_acc, test_loss = model.get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/metaparam_optimization.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/check_gradients.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY, prob_to_class
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string


def evaluate_cost(W, x, y_real, l2_reg):
    model.layers[0].weights = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")
    W = copy.deepcopy(model.layers[0].weights)

    no 	= 	W.shape[0]
    d 	= 	x.shape[0]

    # c = evaluate_cost(W, x, y_real)
    grad_W = np.zeros(W.shape)
    for i in tqdm(range(W.shape[0])):
        for j in range(W.shape[1]):
            W_try = np.matrix(W)
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            W_try = np.matrix(W)
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W

if __name__ == "__main__":
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0:20]
    y = y_train[:, 0:20]
    reg = 0.1

    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    anal_time = time.time()
    model.fit(x, y, batch_size=None, epochs=1, lr=0, # 0 lr will not change weights
                    momentum=0, l2_reg=reg)
    analytical_grad = model.layers[0].gradient
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, l2_reg=reg, h=1e-5)
    num_time = num_time - time.time()

    _EPS = 0.0000001
    denom = np.abs(analytical_grad) + np.abs(numerical_grad)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")
    print("Speedup:", (num_time/anal_time))



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/l2_reg_good.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/lr_ranges.py contains:
 #####################################################
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
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    ns = 800

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-7, lr_max=1e-2, ns=ns)  # Modifies lr while training
    callbacks = [mt, lrs]

    # Fit model
    iterations = 6*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, iterations=iterations,
            l2_reg=10**-1.85, shuffle_minibatch=True,
            callbacks=callbacks)
    # model.save("models/yes_dropout_test")
    
    # # Test model
    val_acc = model.get_metric_loss(x_val, y_val)[0]
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "Test acc: " + str(test_acc)
    mt.plot_training_progress(show=True, save=True, name="figures/lr_limits/final_train", subtitle=subtitle)
    # mt.save("limits_test")
    # lrs = np.load("limits_test_lr.npy")
    # plt.plot(lrs)
    # plt.show()

    # mt.plot_acc_vs_lr(show=True, save=True, name="figures/lr_limits/lr_test", subtitle="")


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/lr_search_plot.py contains:
 #####################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("l2reg_optimization/evaluations.csv")

lr = df[["l2_reg"]].to_numpy()
values = df[["value"]].to_numpy()

plt.scatter(lr, values)
plt.xlabel("l2 regularization")
plt.ylabel("Top Validation Accuracy")
plt.title("Gaussian Process Regression Optimization Evaluations")
plt.show()

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/dropout_test.py contains:
 #####################################################
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

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/joint_code.py contains:
 #####################################################
#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/test_gp_search.py contains:
 #####################################################
import numpy as np
from skopt.space import Real, Integer, Categorical
from gaussian_process import GaussianProcessSearch

from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
import json


a_space = Real(name='lr', low=0., high=1.)
b_space = Integer(name='batch_size', low=0, high=200)
c_space = Real(name='alpha', low=0, high=1)
d_space = Real(name='some_param', low=0, high=100)

search_space = [a_space, b_space, c_space, d_space]
fixed_space = {'noise_level': 0.1}


def func(lr, batch_size, alpha, some_param, noise_level):
    # Max = 101
    return lr**3 + batch_size**2 + some_param * alpha + np.random.randn() * noise_level


gp_search = GaussianProcessSearch(search_space=search_space,
                                  fixed_space=fixed_space,
                                  evaluator=func,
                                  input_file=None,  # Use None to start from zero
                                  output_file='test.csv')
gp_search.init_session()
x, y = gp_search.get_maximum(n_calls=10, n_random_starts=0,
                             noise=fixed_space['noise_level'],
                             verbose=True,
                             )

x = gp_search.get_next_candidate(n_points=5)
print('NEXT CANDIDATES: ' + str(x))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/load_save.py contains:
 #####################################################
import os
import json
import pandas


def load(data_file):
    """Load the given data file and returns a dictionary of the values

    Args:
        data_file (str): Path to a file containing data in one of the known formats (json, csv)

    Returns:
        A dictionary with the loaded data

    """
    _, ext = os.path.splitext(data_file)
    try:
        return known_file_types[ext]['load'](data_file)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def save(data_file, dictionary):
    """Save the given dictionary in the given file. Format is determined by data_file extension

    Args:
        data_file (str): Path to a file in which to save the data. Extension is used to determine
        the format, therefore the path must contain an extension.
        dictionary (dict): Dictionary with the data

    """
    _, ext = os.path.splitext(data_file)
    try:
        known_file_types[ext]['save'](data_file, dictionary)
    except KeyError:
        raise Exception('Error loading file: type ' + str(ext) + ' is not supported')


def _load_json(data_file):
    with open(data_file, 'r') as file:
        return json.load(file)


def _save_json(data_file, dictionary):
    with open(data_file, 'w') as file:
        json.dump(dictionary, file)


def _load_csv(data_file):
    data_frame = pandas.read_csv(data_file, header=0)
    res = {}
    for k in data_frame:
        res[k] = data_frame[k].tolist()
    return res


def _save_csv(data_file, dictionary):
    data_frame = pandas.DataFrame.from_dict(data=dictionary, orient='columns')
    data_frame.to_csv(data_file, header=True, index=False)


known_file_types = {
    '.json':
        {
            'load': _load_json,
            'save': _save_json
        },
    '.csv':
        {
            'load': _load_csv,
            'save': _save_csv
        }
}



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/__init__.py contains:
 #####################################################


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/gaussian_process.py contains:
 #####################################################
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from skopt.plots import plot_objective, plot_evaluations
from skopt import dump, load
import time
import matplotlib.pyplot as plt
import load_save
import sys
import pathlib
import numpy as np

# Session variables
session_params = {}


class GaussianProcessSearch:

    def __init__(self, search_space, fixed_space, evaluator, input_file=None, output_file=None):
        """Instantiate the GaussianProcessSearch and create the GaussianProcessRegressor

        Args:
            search_space (list): List of skopt.space.Dimension objects (Integer, Real,
                or Categorical) whose name must match the correspondent variable name in the
                evaluator function
            fixed_space (dict): Dictionary of parameters that will be passed by default to the
                evaluator function. The keys must match the correspondent names in the function.
            evaluator (function): Function of which we want to estimate the maximum. It must take
                the union of search_space and fixed_space as parameters and return a scalar value.
            input_file (str): Path to the file containing points in the search space and
                corresponding values that are already known.
            output_file (str): Path to the file where updated results will be stored.
        """
        self.search_space = search_space
        self.fixed_space = fixed_space
        self.evaluator = evaluator
        self.input_file = input_file
        self.output_file = output_file
        self.x_values = []
        self.y_values = []
        if input_file is not None:
            try:
                data_dict = load_save.load(data_file=input_file)
                self.x_values, self.y_values = self._extract_values(data_dict)
            except OSError as e:
                raise OSError('Cannot read input file. \n' + str(e))

    @staticmethod
    def _get_gp_regressor(length_scale=1., nu=2.5, noise=0.1):
        """Creates the GaussianProcessRegressor model

        Args:
            length_scale (Union[float, list]): Length scale of the GP kernel. If float, it is the
                same for all dimensions, if array each element defines the length scale of the
                dimension
            nu (float): Controls the smoothness of the approximation.
                see https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

        Returns:
            A skopt.learning.GaussianProcessRegressor with the given parameters

        """
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu)
        return GaussianProcessRegressor(kernel=kernel, alpha=noise ** 2)

    def get_maximum(self, n_calls=10, n_random_starts=5, noise=0.01, verbose=True,
                    plot_results=False):
        """Performs Bayesian optimization by iteratively evaluating the given function on points
        that are likely to be a global maximum.

        After the optimization, the evaluated values are stored in self.x_values and
        self.y_values and appended to the data file if provided.

        Args:
            n_calls (int): Number of iterations
            n_random_starts (int): Initial random evaluations if no previpus values are provided
            noise (float): Estimated noise in the data
            verbose (bool): Whether to print optimization details at each evaluation
            plot_results (bool): Whether to plot an analysis of the solution

        Returns:
            A tuple (x, y) with the argmax and max found of the evaluated function.

        """
        x_values = [x for x in self.x_values] if len(self.x_values) > 0 else None
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values] if len(self.y_values) > 0 else None
        rand_starts = 2 if len(self.x_values) == 0 and n_random_starts == 0 else n_random_starts
        res = gp_minimize(func=GaussianProcessSearch.evaluate,
                          dimensions=self.search_space,
                          n_calls=n_calls,
                          n_random_starts=rand_starts,
                          acq_func='EI',
                          acq_optimizer='lbfgs',
                          x0=x_values,
                          y0=y_values,
                          noise=noise,
                          n_jobs=-1,
                          callback=self.__save_res,
                          verbose=verbose)
        if plot_results:
            ax = plot_objective(res)
            plt.show()
            ax = plot_evaluations(res)
            plt.show()

        self.x_values = [[float(val) for val in point] for point in res.x_iters]
        self.y_values = [-val for val in res.func_vals]
        if self.output_file is not None:
            self.save_values()
            try:
                ax = plot_objective(res)
                plt.savefig( self.output_file + "_objective_plot.png")
            except Exception as e:
                print(e)
            try:
                ax = plot_evaluations(res)
                plt.savefig( self.output_file + "_evaluations_plot.png")
            except Exception as e:
                print(e)
        return res.x, -res.fun

    def add_point_value(self, point, value):
        """Add a point and the correspondent value to the knowledge.

        Args:
            point (Union[list, dict]): List of values correspondent to self.search_space
                dimensions (in the same order), or dictionary {dimension_name: value} for all
                the dimensions in self.search_space.
            value (float): Value of the function at the given point

        """
        p = []
        if isinstance(point, list):
            p = point
        elif isinstance(point, dict):
            for dim in self.search_space:
                p.append(point[dim.name])
        else:
            raise ValueError('Param point of add_point_value must be a list or a dictionary.')
        self.x_values.append(p)
        self.y_values.append(value)

    def get_next_candidate(self, n_points):
        """Returns the next candidates for the skopt acquisition function

        Args:
            n_points (int): Number of candidates desired

        Returns:
            List of points that would be chosen by gp_minimize as next candidate

        """
        # Negate y_values because skopt performs minimization instead of maximization
        y_values = [-y for y in self.y_values]
        optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator='gp',
            n_initial_points=len(self.x_values),
            acq_func='EI'
        )
        optimizer.tell(self.x_values, y_values)  # TODO Does this fit the values???
        points = optimizer.ask(n_points=n_points)
        return self._to_dict_list(points)

    def get_random_candidate(self, n_points):
        candidates = []
        for _ in range(n_points):
            candidate = {}
            for elem in self.search_space:
                candidate[str(elem.name)] = elem.rvs(n_samples=1)[0]
            candidates.append(candidate)
        return candidates

    def _to_dict_list(self, points):
        """Transform the list of points in a list of dictionaries {dimension_name: value}

        Args:
            points (list): List of lists of value, where for each list, the i-th element
            corresponds to a value for the i-th dimension of the search space

        Returns:
            A list of dictionaries, where each dictionary has the search space dimensions as keys
            and the correspondent value of points, in the self.search_space order

        """
        def to_dict(point):
            d = {}
            for i, dim in enumerate(self.search_space):
                d[dim.name] = point[i]
            return d
        return [to_dict(p) for p in points]

    def init_session(self):
        """Save in session variables. the parameters that will be passed to the evaluation function
        by default.

        """
        global session_params
        session_params['fixed_space'] = self.fixed_space
        session_params['evaluator'] = self.evaluator
        session_params['dimension_names'] = [dim.name for dim in self.search_space]

    def reset_session(self):
        """Reset session variables.

        """
        global session_params
        session_params = {}

    def _extract_values(self, data_dict):
        """Extracts the x values and target values from the given data dictionary.

         Args:
             data_dict (dict): A dictionaty like: {<param_name>: [list of values]} where all lists
                 have the same length and values at same index belong to the same point. The only
                 exception is data_dict['value'] that must contain a list of float correspondent
                 to the function evaluations in the points.

         Returns:
             A tuple (x_values, y_values) where
                 x_values (list): List of points in the search space
                 y_values (list): List of known values for the x_values points

        """
        y_values = data_dict['value']
        x_values = []
        for i, dimension in enumerate(self.search_space):
            name = dimension.name
            try:
                for j, v in enumerate(data_dict[name]):
                    if i == 0:  # If first dimension, instantiate an array for data point
                        x_values.append([])
                    x_values[j].append(data_dict[name][j])
            except KeyError:
                raise KeyError('Search space expects a ' + name + ' dimension but loaded data '
                                                                  'does not contain it')
        return x_values, y_values

    def _pack_values(self):
        """Packs the known values to a dictionary where keys are dimension names

        Returns: A dictionary {dimension_name: [dimension_values] for all dimensions,
            value: [result_values]}

        """
        res_dict = {}
        for i, dimension in enumerate(self.search_space):
            res_dict[dimension.name] = []
            for point in self.x_values:
                res_dict[dimension.name].append(point[i])
        res_dict['value'] = self.y_values
        return res_dict

    def save_values(self):
        """Save in the data file the known x_values and y_values

        """
        data_dict = self._pack_values()
        load_save.save(self.output_file, data_dict)

    @staticmethod
    def _to_key_value(values):
        """Transform the given list of values in a key-value dictionary from the search_space names

        Args:
            values (list): List of values of the same length as self.search_space

        Returns:
            A dictionary key[i]: value[i] where key[i] is the name of the i-th dimension of
            self.search_space and value[i] is the i-th element of values

        """
        global session_params
        name_value_dict = {}
        for i, name in enumerate(session_params['dimension_names']):
            name_value_dict[name] = values[i]
        return name_value_dict

    @staticmethod
    def evaluate(point):
        """Evaluate the evaluator function at the given point

        Args:
            point (list): List of values each one corresponding to a dimension of self.search_space

        Returns:
            The value of self.evaluator at the given point, negated (to be used in minimization)
        """
        global session_params
        evaluator_func = session_params['evaluator']
        fixed_space = session_params['fixed_space']
        # Transform the point in a mapping param_name=value
        name_value_dict = GaussianProcessSearch._to_key_value(point)
        args = {**fixed_space, **name_value_dict}
        return -evaluator_func(**args)

    def __save_res(self, res):
        t = time.time()
        pathlib.Path("gpro_results/").mkdir(parents=True, exist_ok=True)
        result_name = "gpro_results/" + str(t) + "_gpro_result.pkl" 
        dump(res, result_name)
        numpy_name = "gpro_results/" + str(t) + "_gpro_res.npy" 
        np.save(numpy_name, res.x)
        numpy_name = "gpro_results/" + str(t) + "_gpro_fun.npy" 
        np.save(numpy_name, -res.fun)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/HyperParameter-Optimizer/metaparamoptimizer.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import itertools as it
import numpy as np
import pickle

# TODO(Oleguer): Think about the structure of all this

class MetaParamOptimizer:
    def __init__(self, save_path=""):
        self.save_path = save_path  # Where to save best result and remaining to explore
        pass

    def list_search(self, evaluator, dicts_list, fixed_args):
        """ Evaluates model (storing best) on provided list of param dictionaries
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        max_result = None
        for indx, evaluable_args in enumerate(dicts_list):
            print("MetaParamOptimizer evaluating:", indx, "/", len(dicts_list), ":", evaluable_args)
            args = {**evaluable_args, **fixed_args}  # Merge kwargs and evaluable_args dicts
            try:
                result = evaluator(**args)
            except Exception as e:
                print("MetaParamOptimizer: Exception found when evaluating:")
                print(e)
                print("Skipping to next point...")
                continue
            if (max_result is None) or (result["value"] > max_result["value"]):
                max_result = result
                max_result["best_params"] = evaluable_args
                self.save(max_result, name="metaparam_search_best_result")  # save best result found so far
            # Save remaning tests (in case something goes wrong, know where to keep testing)
            self.save(dicts_list[indx+1:], name="remaining_tests")
        return max_result

    def grid_search(self, evaluator, search_space, fixed_args):
        """ Performs grid search on specified search_space
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        points_to_evaluate = self.__get_all_dicts(search_space)
        return self.list_search(evaluator, points_to_evaluate, fixed_args)

    def GPR_optimizer(self, evaluator, search_space, fixed_args):
        pass # The other repo

    def save(self, elem, name="best_result"):
        """ Saves result to disk"""
        with open(self.save_path + "/" + name + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, name="best_model", path=None):
        if path is None:
            path = self.save_path
        with open(path + "/" + name, 'rb') as input:
            remaining_tests = pickle.load(input)
        return remaining_tests

    def __get_all_dicts(self, param_space):
        """ Given:
            dict of item: list(elems)
            returns:
            list (dicts of item : elem)
        """
        allparams = sorted(param_space)
        combinations = it.product(*(param_space[Name] for Name in allparams))
        dictionaries = []
        for combination in combinations:
            dictionary = {}
            for indx, name in enumerate(allparams):
                dictionary[name] = combination[indx]
            dictionaries.append(dictionary)
        return dictionaries

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/cyclical_lr_test.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.layers import Dense, Softmax, Relu
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
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    ns = 500

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    bms = BestModelSaver(save_dir=None)  # Saves model with highest val_metric
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  # Modifies lr while training
    callbacks = [mt, bms, lrs]

    # Fit model
    iterations = 2*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=100, epochs=None, iterations=iterations, lr=0.01, momentum=0.0,
                        l2_reg=0.01, shuffle_minibatch=False,
                        callbacks=callbacks)
    # model.save("models/mlp_overfit_test")
    mt.plot_training_progress(show=False, save=True, name="figures/mlp_cyclic_good")
    mt.plot_lr_evolution(show=False, save=True, name="figures/lr_cyclic_good")
    
    # Test model
    best_model = bms.get_best_model()
    test_acc, test_loss = best_model.get_metric_loss(x_test, y_test)
    print("Test accuracy:", test_acc)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/l2reg_search.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/cyclical_lr_good.py contains:
 #####################################################
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/HyperParameter-Optimizer/")

from skopt.space import Real
from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Dense, Softmax, Relu
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from util.misc import dict_to_string

from gaussian_process import GaussianProcessSearch

def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", **kwargs):
    # Saving directories
    figure_file = "figures/" + experiment_name + "/" + dict_to_string(kwargs)
    model_file = "models/" + experiment_name + "/" + dict_to_string(kwargs)

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    # Pick metaparams
    batch_size = 100
    ns = 2*np.floor(x_train.shape[1]/batch_size)
    iterations = 4*ns  # 2 cycles

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    # bms = BestModelSaver(save_dir=None)
    lrs = LearningRateScheduler(
        evolution="cyclic", lr_min=1e-5, lr_max=1e-1, ns=ns)  
    # callbacks = [mt, bms, lrs]
    callbacks = [mt, lrs]

    # Adjust logarithmic
    kwargs["l2_reg"] = 10**kwargs["l2_reg"]

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=batch_size, epochs=None, iterations=iterations, **kwargs,
              callbacks=callbacks)

    # Write results    
    # best_model = bms.get_best_model()
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", Test Acc: " + str(test_acc)
    mt.plot_training_progress(show=False, save=True, name=figure_file, subtitle=subtitle)

    # Maximizing value: validation accuracy
    # val_metric = bms.best_metric
    val_metric = model.get_metric_loss(x_val, y_val)[0]
    return val_metric


if __name__ == "__main__":
    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    for i in [2, 3, 4, 5]:
        x, y = LoadXY("data_batch_" + str(i))
        x_train = np.concatenate((x_train, x), axis=1)
        y_train = np.concatenate((y_train, y), axis=1)
    x_val = x_train[:, -5000:]
    y_val = y_train[:, -5000:]
    x_train = x_train[:, :-5000]
    y_train = y_train[:, :-5000]
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    fixed_args = {
        "experiment_name": "l2reg_optimization",
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "momentum": 0.0,
        "shuffle_minibatch":False,
    }

    pathlib.Path(fixed_args["experiment_name"]).mkdir(parents=True, exist_ok=True)
    l2reg_space = Real(name='l2_reg', low=-7, high=-1)
    search_space = [l2reg_space]

    gp_search = GaussianProcessSearch(search_space=search_space,
                                      fixed_space=fixed_args,
                                      evaluator=evaluator,
                                      input_file=None,
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls=15,
                                 n_random_starts=7,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/overfit_test.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY

np.random.seed(0)

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

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
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    # Fit model
    x_train = x_train[:, 0:100]
    y_train = y_train[:, 0:100]
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                        batch_size=100, epochs=200, lr=0.001, momentum=0.0,
                        l2_reg=0.0, shuffle_minibatch=False, save_path="models/mlp_overfit_test")
    model.plot_training_progress(save=True, name="figures/mlp_overfit_test")
    model.save("models/mlp_overfit_test")

    # Test model
    test_acc, test_loss = model.get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/metaparam_optimization.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/check_gradients.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY, prob_to_class
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string


def evaluate_cost(W, x, y_real, l2_reg):
    model.layers[0].weights = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")
    W = copy.deepcopy(model.layers[0].weights)

    no 	= 	W.shape[0]
    d 	= 	x.shape[0]

    # c = evaluate_cost(W, x, y_real)
    grad_W = np.zeros(W.shape)
    for i in tqdm(range(W.shape[0])):
        for j in range(W.shape[1]):
            W_try = np.matrix(W)
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            W_try = np.matrix(W)
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W

if __name__ == "__main__":
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0:20]
    y = y_train[:, 0:20]
    reg = 0.1

    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    anal_time = time.time()
    model.fit(x, y, batch_size=None, epochs=1, lr=0, # 0 lr will not change weights
                    momentum=0, l2_reg=reg)
    analytical_grad = model.layers[0].gradient
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, l2_reg=reg, h=1e-5)
    num_time = num_time - time.time()

    _EPS = 0.0000001
    denom = np.abs(analytical_grad) + np.abs(numerical_grad)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad-numerical_grad),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")
    print("Speedup:", (num_time/anal_time))



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/l2_reg_good.py contains:
 #####################################################
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


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/lr_ranges.py contains:
 #####################################################
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
    model.add(Dense(nodes=50, input_dim=x_train.shape[0]))
    model.add(Relu())
    model.add(Dense(nodes=10, input_dim=50))
    model.add(Softmax())

    ns = 800

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info
    lrs = LearningRateScheduler(evolution="cyclic", lr_min=1e-7, lr_max=1e-2, ns=ns)  # Modifies lr while training
    callbacks = [mt, lrs]

    # Fit model
    iterations = 6*ns
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
            batch_size=100, iterations=iterations,
            l2_reg=10**-1.85, shuffle_minibatch=True,
            callbacks=callbacks)
    # model.save("models/yes_dropout_test")
    
    # # Test model
    val_acc = model.get_metric_loss(x_val, y_val)[0]
    test_acc = model.get_metric_loss(x_test, y_test)[0]
    subtitle = "Test acc: " + str(test_acc)
    mt.plot_training_progress(show=True, save=True, name="figures/lr_limits/final_train", subtitle=subtitle)
    # mt.save("limits_test")
    # lrs = np.load("limits_test_lr.npy")
    # plt.plot(lrs)
    # plt.show()

    # mt.plot_acc_vs_lr(show=True, save=True, name="figures/lr_limits/lr_test", subtitle="")


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/lr_search_plot.py contains:
 #####################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("l2reg_optimization/evaluations.csv")

lr = df[["l2_reg"]].to_numpy()
values = df[["value"]].to_numpy()

plt.scatter(lr, values)
plt.xlabel("l2 regularization")
plt.ylabel("Top Validation Accuracy")
plt.title("Gaussian Process Regression Optimization Evaluations")
plt.show()

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_2/dropout_test.py contains:
 #####################################################


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/join_code.py contains:
 #####################################################
from glob import glob
import os

files = [f for f in glob('/home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/**', recursive=True) if os.path.isfile(f)]
files = [f for f in files if ".py" in f and ".pyc" not in f and "/examples/" not in f and "Assignment_1" not in f]

with open("joint_code.py", 'wb') as list_file:
    for file in files:
        with open(file, 'rb') as f:
            f_content = f.read()
            list_file.write(("#####################################################\n").encode('utf-8'))
            list_file.write(('# The file %s contains:\n ' % file).encode('utf-8'))
            list_file.write(("#####################################################\n").encode('utf-8'))
            list_file.write(f_content)
            list_file.write(b'\n')
            list_file.write(b'\n')

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mpo/metaparamoptimizer.py contains:
 #####################################################
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import itertools as it
import numpy as np
import pickle

# TODO(Oleguer): Think about the structure of all this

class MetaParamOptimizer:
    def __init__(self, save_path=""):
        self.save_path = save_path  # Where to save best result and remaining to explore
        pass

    def list_search(self, evaluator, dicts_list, fixed_args):
        """ Evaluates model (storing best) on provided list of param dictionaries
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        max_result = None
        for indx, evaluable_args in enumerate(dicts_list):
            print("MetaParamOptimizer evaluating:", indx, "/", len(dicts_list), ":", evaluable_args)
            args = {**evaluable_args, **fixed_args}  # Merge kwargs and evaluable_args dicts
            try:
                result = evaluator(**args)
            except Exception as e:
                print("MetaParamOptimizer: Exception found when evaluating:")
                print(e)
                print("Skipping to next point...")
                continue
            if (max_result is None) or (result["value"] > max_result["value"]):
                max_result = result
                max_result["best_params"] = evaluable_args
                self.save(max_result, name="metaparam_search_best_result")  # save best result found so far
            # Save remaning tests (in case something goes wrong, know where to keep testing)
            self.save(dicts_list[indx+1:], name="remaining_tests")
        return max_result

    def grid_search(self, evaluator, search_space, fixed_args):
        """ Performs grid search on specified search_space
            running evaluator(**kwargs = fixed_args + sample(search_space))
            evaluator should return a dictionary conteining (at least) the field "value" to maximize
            returns result of maximum result["value"] reached adding result["best_params"] that obtained it
        """
        points_to_evaluate = self.__get_all_dicts(search_space)
        return self.list_search(evaluator, points_to_evaluate, fixed_args)

    def GPR_optimizer(self, evaluator, search_space, fixed_args):
        pass # The other repo

    def save(self, elem, name="best_result"):
        """ Saves result to disk"""
        with open(self.save_path + "/" + name + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, name="best_model", path=None):
        if path is None:
            path = self.save_path
        with open(path + "/" + name, 'rb') as input:
            remaining_tests = pickle.load(input)
        return remaining_tests

    def __get_all_dicts(self, param_space):
        """ Given:
            dict of item: list(elems)
            returns:
            list (dicts of item : elem)
        """
        allparams = sorted(param_space)
        combinations = it.product(*(param_space[Name] for Name in allparams))
        dictionaries = []
        for combination in combinations:
            dictionary = {}
            for indx, name in enumerate(allparams):
                dictionary[name] = combination[indx]
            dictionaries.append(dictionary)
        return dictionaries

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/losses.py contains:
 #####################################################
import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    """ Abstact class to represent Loss functions
        An activation has:
            - forward method to apply activation to the inputs
            - backward method to add activation gradient to the backprop
    """
    def __init__(self):
        self._EPS = 1e-5

    @abstractmethod
    def __call__(self, Y_pred, Y_real):
        """ Computes loss value according to predictions and true labels"""
        pass

    @abstractmethod
    def backward(self, Y_pred, Y_real):
        """ Computes loss gradient according to predictions and true labels"""
        pass


# LOSSES IMPLEMNETATIONS  #########################################

class CrossEntropy(Loss):
    def __call__(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0)))/float(Y_pred.shape[1])

    def backward(self, Y_pred, Y_real):
        # d(-log(x))/dx = -1/x
        f_y = np.multiply(Y_real, Y_pred)
        # Element-wise inverse
        loss_diff = - \
            np.reciprocal(f_y, out=np.zeros_like(
                Y_pred), where=abs(f_y) > self._EPS)
        return loss_diff/float(Y_pred.shape[1])


class CategoricalHinge(Loss):
    def __call__(self, Y_pred, Y_real):
        # L = SUM_data (SUM_dim_j(not yi) (MAX(0, y_pred_j - y_pred_yi + 1)))
        pos = np.sum(np.multiply(Y_real, Y_pred),
                     axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        val = neg + 1. - pos
        val = np.multiply(val, (val > 0))
        return np.sum(val)/float(Y_pred.shape[1])

    def backward(self, Y_pred, Y_real):
        # Forall j != yi: (y_pred_j - y_pred_yi + 1 > 0)
        # If     j == yi: -1 SUM_j(not yi) (y_pred_j - y_pred_yi + 1 > 0)
        pos = np.sum(np.multiply(Y_real, Y_pred),
                     axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        wrong_class_activations = np.multiply(
            1-Y_real, (neg + 1. - pos > 0))  # Val of wrong results
        wca_sum = np.sum(wrong_class_activations, axis=0)
        neg_wca = np.einsum("ij,j->ij", Y_real, np.array(wca_sum).flatten())
        return (wrong_class_activations - neg_wca)/float(Y_pred.shape[1])


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/layers.py contains:
 #####################################################
from abc import ABC, abstractmethod
import copy
import numpy as np


class Layer(ABC):
    """ Abstact class to represent Layer layers
        An Layer has:
            - __call__ method to apply Layer function to the inputs
            - gradient method to add Layer function gradient to the backprop
    """
    name = "GenericLayer"

    def __init__(self):
        self.weights = None

    @abstractmethod
    def __call__(self, inputs):
        """ Applies a function to given inputs """
        pass

    @abstractmethod
    def backward(self, in_gradient):
        """ Receives right-layer gradient and multiplies it by current layer gradient """
        pass


# TRAINABLE LAYERS ######################################################

class Dense(Layer):
    def __init__(self, nodes, input_dim, weight_initialization="in_dim"):
        self.nodes = nodes
        self.input_shape = input_dim
        self.__initialize_weights(weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates

    def __call__(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return self.weights*self.inputs

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        # Remove bias TODO Think about this
        left_layer_gradient = (self.weights.T*in_gradient)[:-1, :]

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        # TODO: Rremove self if not going to update it
        self.gradient = in_gradient*self.inputs.T + regularization_term
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "normal":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./100.,
                                    (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.matrix(np.random.normal(
                0.0, 1./float(np.sqrt(self.input_shape)),
                (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape))
            self.weights = np.matrix(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape+1)))  # Add biases


# Activation Layers ######################################################
class Softmax(Layer):
    def __call__(self, x):
        self.outputs = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient


class Relu(Layer):
    def __call__(self, x):
        self.inputs = x
        return np.multiply(x, (x > 0))

    def backward(self, in_gradient, **kwargs):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

class Dropout(Layer):
    def __init__(self, ones_ratio=0.7):
        self.name = "Dropout"
        self.ones_ratio = ones_ratio

    def __call__(self, x, apply=True):
        if apply:
            self.mask = np.random.choice([0, 1], size=(x.shape), p=[1 - self.ones_ratio, self.ones_ratio])
            return np.multiply(self.mask, x)
        return x

    def backward(self, in_gradient, **kwargs):
        return np.multiply(self.mask, in_gradient)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/models.py contains:
 #####################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import copy
from tqdm import tqdm
import pickle

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from utils import minibatch_split

# from callbacks import Callback, LearningRateScheduler


class Sequential:
    def __init__(self, loss=None, pre_trained=None, metric=None):
        self.layers = []

        assert(loss is not None)  # You need a loss!!!
        self.loss = loss
        self.metric = metric

        if pre_trained is not None:
            self.load(pre_trained)

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)

    def predict(self, X, apply_dropout=True):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            if layer.name == "Dropout":
                vals = layer(vals, apply_dropout)
            else:
                vals = layer(vals)
        return vals

    def get_metric_loss(self, X, Y_real, use_dropout=True):
        """ Returns loss and classification accuracy """
        if X is None or Y_real is None:
            print("problem")
            return 0, np.inf
        Y_pred_prob = self.predict(X, use_dropout)
        metric_val = 0
        if self.metric is not None:
            metric_val = self.metric(Y_pred_prob, Y_real)
        loss = self.loss(Y_pred_prob, Y_real)
        return metric_val, loss

    def cost(self, Y_pred_prob, Y_real, l2_reg):
        """Computes cost = loss + regularization"""
        # Loss
        loss_val = self.loss(Y_pred_prob, Y_real)
        # Regularization
        w_norm = 0
        for layer in self.layers:
            if layer.weights is not None:
                w_norm += np.linalg.norm(layer.weights, 'fro')**2
        return loss_val + l2_reg*w_norm

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=None,
            iterations = None, lr=0.01, momentum=0.7, l2_reg=0.1,
            shuffle_minibatch=True, callbacks=[], **kwargs):
        """ Performs backrpop with given parameters.
            save_path is where model of best val accuracy will be saved
        """
        assert(epochs is None or iterations is None) # Only one can set it limit
        if iterations is not None:
            epochs = int(np.ceil(iterations/(X.shape[1]/batch_size)))
        # Store vars as class variables so they can be accessed by callbacks
        # TODO(think a better way)
        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.val_metric = 0
        self.t = 0

        # Call callbacks
        for callback in callbacks:
            callback.on_training_begin(self)

        # Training
        stop = False
        pbar = tqdm(list(range(self.epochs)))
        for self.epoch in pbar:
        # for self.epoch in range(self.epochs):
            for X_minibatch, Y_minibatch in minibatch_split(X, Y, batch_size, shuffle_minibatch):
                Y_pred_prob = self.predict(X_minibatch)  # Forward pass
                gradient = self.loss.backward(
                    Y_pred_prob, Y_minibatch)  # Loss grad
                for layer in reversed(self.layers):  # Backprop (chain rule)
                    gradient = layer.backward(
                        in_gradient=gradient,
                        lr=self.lr,  # Trainable layer parameters
                        momentum=self.momentum,
                        l2_regularization=self.l2_reg)
                # Call callbacks
                for callback in callbacks:
                    callback.on_batch_end(self)
                self.t += 1  # Step counter
                if self.t >= iterations:
                    stop = True
                    break
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(self)
            # Update progressbar
            # pbar.set_description("Val acc: " + str(self.val_metric))
            if stop:
                break


    # IO functions ################################################
    def save(self, path):
        """ Saves current model to disk (Dont put file extension)"""
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        with open(path + ".pkl", 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        """ Loads model to disk (Dont put file extension)"""
        with open(path + ".pkl", 'rb') as input:
            tmp_dict = pickle.load(input)
            self.__dict__.update(tmp_dict)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/callbacks.py contains:
 #####################################################
from abc import ABC, abstractmethod
import copy
import numpy as np
from matplotlib import pyplot as plt

import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
from models import Sequential


class Callback(ABC):
    """ Abstract class to hold callbacks
    """

    def on_training_begin(self, model):
        pass

    def on_batch_end(self, model):
        pass

    # @abstractmethod
    def on_epoch_end(self, model):
        """ Getss called at the end of each epoch
            Can modify model training variables (eg: LR Scheduler)
            Can store information to be retrieved afterwards (eg: Metric Tracker)
        """
        pass


# INFORMATION STORING CALLBACKS ######################################################
class MetricTracker(Callback):
    """ Tracks training metrics to plot and save afterwards
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []

    def on_training_begin(self, model):
        self.metric_name = model.metric.name
        self.__track(model)

    def on_batch_end(self, model):
        # self.learning_rates.append(model.lr)
        # self.__track(model)
        pass

    def on_epoch_end(self, model):
        self.__track(model)
        pass

    def __track(self, model):
        train_metric, train_loss = model.get_metric_loss(model.X, model.Y, use_dropout=False)
        val_metric, val_loss = model.get_metric_loss(model.X_val, model.Y_val, use_dropout=False)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)
        self.learning_rates.append(model.lr)
        model.val_metric = val_metric

    def plot_training_progress(self, show=True, save=False, name="model_results", subtitle=None):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(bottom=np.nanmin(self.val_losses)/2)
        ax1.set_ylim(top=1.25*np.nanmax(self.val_losses))
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='center right')

        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel(self.metric_name)
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=1)
        n = len(self.train_metrics)
        ax2.plot(list(range(n)),
                 np.array(self.train_metrics), label="Train acc", c="green")
        if len(self.val_metrics) > 0:
            n = len(self.val_metrics)
            ax2.plot(list(range(n)),
                     np.array(self.val_metrics), label="Val acc", c="blue")
        ax2.tick_params(axis='y')

        # plt.tight_layout()
        plt.suptitle("Training Evolution")
        if subtitle is not None:
            plt.title(subtitle)
        plt.legend(loc='upper right')

        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def plot_lr_evolution(self, show=True, save=False, name="lr_evolution", subtitle=None):
        plt.suptitle("Learning rate evolution")
        plt.plot(self.learning_rates, label="Learning rate")
        plt.legend(loc='upper right')
        plt.xlabel("Iteration")
        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def plot_acc_vs_lr(self, show=True, save=False, name="acc_vs_lr", subtitle=None):
        plt.suptitle("Accuracy evolution for each learning rate")
        plt.plot(np.log(self.learning_rates), self.train_metrics, label="Accuracy")
        plt.legend(loc='upper right')
        plt.xlabel("Learning Rate")
        plt.ylabel("Train Accuracy")
        if save:
            directory = "/".join(name.split("/")[:-1])
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            plt.savefig(name + ".png")
            plt.close()
        if show:
            plt.show()

    def save(self, file):
        np.save(file + "_lr", self.learning_rates)
        np.save(file + "_acc", self.train_metrics)


class BestModelSaver(Callback):
    def __init__(self, save_dir=None):
        self.save_dir = None
        if save_dir is not None:
            self.save_dir = os.path.join(save_dir, "best_model")
        self.best_metric = -np.inf
        self.best_model_layers = None
        self.best_model_loss = None
        self.best_model_metric = None

    def on_batch_end(self, model):
        val_metric = model.get_metric_loss(model.X_val, model.Y_val)[0]
        if val_metric >= self.best_metric:
            self.best_metric = model.val_metric
            self.best_model_layers = copy.deepcopy(model.layers)
            self.best_model_loss = copy.deepcopy(model.loss)
            self.best_model_metric = copy.deepcopy(model.metric)
            if self.save_dir is not None:
                model.save(self.save_dir)

    def get_best_model(self):
        best_model = Sequential(loss=self.best_model_loss,
                                metric=self.best_model_metric)
        best_model.layers = self.best_model_layers
        return best_model


# LEARNING PARAMS MODIFIER CALLBACKS ######################################################

class LearningRateScheduler(Callback):
    def __init__(self, evolution="linear", lr_min=None, lr_max=None, ns=500):
        assert(evolution in ["constant", "linear", "cyclic"])
        self.type = evolution
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.ns = ns

    def on_training_begin(self, model):
        if self.type == "cyclic":
            model.lr = self.lr_min

    def on_batch_end(self, model):
        if self.type == "cyclic":
            slope = int(model.t/self.ns)%2
            lr_dif = float(self.lr_max - self.lr_min)
            if slope == 0:
                model.lr = self.lr_min + float(model.t%self.ns)*lr_dif/float(self.ns)
            if slope == 1:
                model.lr = self.lr_max - float(model.t%self.ns)*lr_dif/float(self.ns)
        if self.type == "linear":
            lr_dif = float(self.lr_max - self.lr_min)
            model.lr = self.lr_min + float(model.t)*lr_dif/float(self.ns)

    def on_epoch_end(self, model):
        if self.type == "constant":
            pass
        elif self.type == "linear":
            model.lr = 0.9*model.lr

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/utils.py contains:
 #####################################################
import matplotlib.pyplot as plt
import numpy as np
import cv2

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('data/'+filename, 'rb') as fo:
		dictionary = pickle.load(fo, encoding='bytes')
	return dictionary

def getXY(dataset, num_classes=10):
	"""Splits dataset into 2 np mat x, y (dim along rows)"""
	# 1. Convert labels to one-hot vectors
	labels = np.array(dataset[b"labels"])
	one_hot_labels = np.zeros((labels.size, num_classes))
	one_hot_labels[np.arange(labels.size), labels] = 1
	return np.mat(dataset[b"data"]).T, np.mat(one_hot_labels).T

def LoadXY(filename):
	return getXY(LoadBatch(filename))

def plot(flatted_image, shape=(32, 32, 3), order='F'):
	image = np.reshape(flatted_image, shape, order=order)
	cv2.imshow("image", image)
	cv2.waitKey()

def accuracy(Y_pred_classes, Y_real):
	return np.sum(np.multiply(Y_pred_classes, Y_real))/Y_pred_classes.shape[1]

def minibatch_split(X, Y, batch_size, shuffle=True):
	"""Yields splited X, Y matrices in minibatches of given batch_size"""
	if (batch_size is None) or (batch_size > X.shape[1]):
		batch_size = X.shape[1]
	indx = list(range(X.shape[1]))
	if shuffle:
		np.random.shuffle(indx)
	for i in range(int(X.shape[1]/batch_size)):
		pos = i*batch_size
		# Get minibatch
		X_minibatch = X[:, indx[pos:pos+batch_size]]
		Y_minibatch = Y[:, indx[pos:pos+batch_size]]
		if i == int(X.shape[1]/batch_size) - 1:  # Get all the remaining
			X_minibatch = X[:, indx[pos:]]
			Y_minibatch = Y[:, indx[pos:]]
		yield X_minibatch, Y_minibatch

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/metrics.py contains:
 #####################################################
import numpy as np

class Accuracy:
	def __init__(self):
		self.name = "Accuracy"

	def __call__(self, Y_pred, Y_real):
		Y_pred_class = self.__prob_to_class(Y_pred)
		return np.sum(np.multiply(Y_pred_class, Y_real))/Y_pred_class.shape[1]


	def __prob_to_class(self, Y_pred_prob):
		"""Given array of prob, returns max prob in one-hot fashon"""
		idx = np.argmax(Y_pred_prob, axis=0)
		Y_pred_class = np.zeros(Y_pred_prob.shape)
		Y_pred_class[idx, np.arange(Y_pred_class.shape[1])] = 1
		return Y_pred_class


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/util/misc.py contains:
 #####################################################

def dict_to_string(dictionary):
    s = str(dictionary)
    s = s.replace(" ", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("'", "")
    s = s.replace(":", "-")
    s = s.replace(",", "_")
    return s

