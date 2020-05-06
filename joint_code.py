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
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/overfit_test.py contains:
 #####################################################

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

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=200, n_val=200, n_test=2)

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=64, kernel_shape=(4, 4), input_shape=(28, 28, 1)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2)))
    model.add(Flatten())
    model.add(Dense(nodes=400))
    model.add(Relu())
    model.add(Dense(nodes=10))
    model.add(Softmax())

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=200, lr = 1e-2, momentum=0.5, callbacks=callbacks)
    model.save("models/mnist_test_conv_2")

    mt.plot_training_progress()
    y_pred_prob = model.predict(x_train)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/check_gradients.py contains:
 #####################################################

import numpy as np
import sys, pathlib
from helper import read_mnist, read_cifar_10, read_names
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

np.random.seed(1)

def evaluate_cost_W(W, x, y_real, l2_reg, filter_id):
    model.layers[0].filters[filter_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def evaluate_cost_b(W, x, y_real, l2_reg, bias_id):
    model.layers[0].biases[bias_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")

    grads_w = []
    for filter_id, filt in enumerate(model.layers[0].filters):
        W = copy.deepcopy(filt)  # Compute W
        grad_W = np.zeros(W.shape)
        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                for c in range(W.shape[2]):
                    W_try = np.array(W)
                    # print(W_try.shape)
                    W_try[i,j,c] -= h
                    c1 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    W_try = np.array(W)
                    W_try[i,j,c] += h
                    c2 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    grad_W[i,j,c] = (c2-c1) / (2*h)

                    model.layers[0].filters[filter_id] = W  # Reset it

        grads_w.append(grad_W)

    grads_b = []
    for bias_id, bias in enumerate(model.layers[0].biases):
        b_try = copy.deepcopy(bias) - h
        c1 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        b_try = copy.deepcopy(bias) + h
        c2 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        grad_b = (c2-c1) / (2*h)
        model.layers[0].biases[bias_id] = bias  # Reset it

        grads_b.append(grad_b)
    return grads_w, grads_b

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=3, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=2, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_names(n_train=500)

    class_sum = np.sum(y_train, axis=1)*y_train.shape[0]
    class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)

    print(class_count)

    print(type(x_train[0, 0, 0]))

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=2, kernel_shape=(4, 4), stride=3, dilation_rate=2, input_shape=x_train.shape[0:-1]))
    model.add(Relu())
    model.add(MaxPool2D((2, 2), stride=3))
    model.add(Flatten())
    model.add(Dense(nodes=y_train.shape[0]))
    model.add(Relu())
    model.add(Softmax())

    print(np.min(np.abs(model.layers[0].filters)))

    reg = 0.0

    # Fit model
    anal_time = time.time()
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=200, epochs=1, lr=0, momentum=0, l2_reg=reg)
    analytical_grad_weight = model.layers[0].filter_gradients
    analytical_grad_bias = model.layers[0].bias_gradients
    # print(analytical_grad_weight)
    print(analytical_grad_bias)
    anal_time = time.time() - anal_time

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad_w, numerical_grad_b = ComputeGradsNum(x_train, y_train, model, l2_reg=reg, h=1e-5)
    # print(numerical_grad_w)
    print(numerical_grad_b)
    num_time = time.time() - num_time

    print("Weight Error:")
    _EPS = 0.0000001
    denom = np.abs(analytical_grad_weight) + np.abs(numerical_grad_w)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")


    print("Bias Error:")
    _EPS = 0.000000001
    denom = np.abs(analytical_grad_bias) + np.abs(numerical_grad_b)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")

    print("Speedup:", (num_time/anal_time))



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/cifar_10_test.py contains:
 #####################################################

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
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=30000, n_val=200, n_test=200)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10()

    print(x_train.shape)
    # print(y_train.shape)

    # for i in range(200):
    #     cv2.imshow("image", x_train[..., i])
    #     cv2.waitKey()

    # Define callbacks
    mt = MetricTracker(file_name="cifar_test_3")  # Stores training evolution info (losses and metrics)
    # bms = BestModelSaver("models/best_cifar")  # Stores training evolution info (losses and metrics)
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
    # model.load("models/cifar_test_2")
    # mt.load("models/tracker")
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=20, momentum=0.9, l2_reg=0.003, callbacks=callbacks)
    model.save("models/cifar_test_3")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    # y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/names.py contains:
 #####################################################

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
    n1 = 40  # Filters of first Conv2D
    k1 = 6   # First kernel y size
    n2 = 20  # Filters of second Conv2D
    k2 = 4   # Second kernel y size
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
              batch_size=100, epochs=500, lr = 1e-3, momentum=0.8, l2_reg=0.001,
              compensate=True, callbacks=callbacks)
    model.save("models/names_best")

    mt.plot_training_progress(save=True, name="figures/names_best")
    # y_pred_prob = model.predict(x_train)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/confusion_matrices.py contains:
 #####################################################

import numpy as np
import sys, pathlib
from helper import read_names, read_names_countries
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import plot_confusion_matrix

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, _, _ = read_names(n_train=-1)
    print(x_train.shape)
    classes = read_names_countries()

    # Load model model
    model = Sequential(loss=CrossEntropy())
    # model.load("models/names_test")
    model.load("models/name_metaparam_search_2/n1-39_n2-33_k1-2_k2-10_batch_size-50")

    y_pred_train = model.predict_classes(x_train)
    y_pred_val = model.predict_classes(x_val)
    
    plot_confusion_matrix(y_pred_train, y_train, classes, "figures/conf_best_model")
    plot_confusion_matrix(y_pred_val, y_val, classes, "figures/conf_best_model_val")


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/metaparam_search.py contains:
 #####################################################
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
        "experiment_name": "name_metaparam_search_2",
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
                                      input_file="name_metaparam_search" + '/evaluations.csv',
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls = 12,
                                 n_random_starts=0,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/helper.py contains:
 #####################################################
import numpy as np

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
    
    filename = "data/" + filename
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data
    
def read_mnist(dim=[28,28],n_train=50000, n_val=10000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    
    import scipy.misc

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    def rs(imgs):
        imgs = (imgs.T).reshape((dim[0], dim[1], imgs.shape[0]), order='C')
        return np.expand_dims(imgs, axis=2).astype(float)  # h, w, c, n
    return rs(train_imgs[:n_train]),train_lbls_1hot[:n_train].T.astype(float),\
           rs(train_imgs[n_train:n_train+n_val]),train_lbls_1hot[n_train:n_train+n_val].T.astype(float),\
           rs(test_imgs[:n_test]),test_lbls_1hot[:n_test].T.astype(float)

def read_cifar_10(n_train=None, n_val=None, n_test=None):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
    from mlp.utils import LoadXY

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

    return x_train.astype(float), y_train.astype(float), x_val.astype(float), y_val.astype(float), x_test.astype(float), y_test.astype(float)

def read_names(n_train=-1):
    def read_file(filepath="data/names/ascii_names.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = -1
        for name in names:
            n_len = max(n_len, len(name))
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)

    val_indxs = []
    with open("data/names/Validation_Inds.txt") as fp:
        val_indxs = [int(val) for val in fp.readline().split(" ")]

    indx = list(range(len(names)))
    np.random.shuffle(indx)
    for val in val_indxs:
        indx.remove(val)

    return x[..., indx[:n_train]], y[..., indx[:n_train]],\
           x[..., val_indxs], y[..., val_indxs],\
           None, None


def read_names_test(n_train=-1):
    def read_file(filepath="data/names/test.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = 19
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)
    return x, y, names

def read_names_countries():
    return ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German",\
            "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese",\
            "Russian", "Scottish", "Spanish", "Vietnamese"]

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/test.py contains:
 #####################################################
import numpy as np
import sys, pathlib
import cv2

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.callbacks import MetricTracker

mt = MetricTracker()
mt.metric_name = "Accuracy"
mt.load("models/tracker")
mt.plot_training_progress()


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/mnist.py contains:
 #####################################################

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

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=200, n_val=200, n_test=2)

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=64, kernel_shape=(4, 4), input_shape=(28, 28, 1)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2)))
    # model.add(Conv2D(num_filters=32, kernel_shape=(3, 3)))
    # model.add(Relu())
    model.add(Flatten())
    # model.add(Flatten(input_shape=(28, 28, 1)))
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
              batch_size=100, epochs=200, lr = 1e-2, momentum=0.5, callbacks=callbacks)
    model.save("models/mnist_test_conv_2")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/test_name_model.py contains:
 #####################################################

import numpy as np
import sys, pathlib
from helper import read_names, read_names_countries, read_names_test
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import plot_confusion_matrix

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_test, y_test, names = read_names_test()
    classes = read_names_countries()

    print(x_test.shape)

    # Load model model
    model = Sequential(loss=CrossEntropy())
    model.load("models/names_test")
    # model.load("models/names_no_compensation")

    y_pred_prob_test = model.predict(x_test)
    y_pred_test = model.predict_classes(x_test)
    print(y_pred_prob_test)
    print(y_test)
    
    plot_confusion_matrix(y_pred_test, y_test, classes, "figures/conf_test")

    import matplotlib.pyplot as plt
    plt.title("Prediction Vectors")      
    pos = plt.imshow(y_pred_prob_test.T)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    # plt.xticks(rotation=45, ha='right')
    plt.colorbar(pos)
    plt.savefig("figures/prob_vector_test")
    plt.show()

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
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/overfit_test.py contains:
 #####################################################

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

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=200, n_val=200, n_test=2)

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=64, kernel_shape=(4, 4), input_shape=(28, 28, 1)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2)))
    model.add(Flatten())
    model.add(Dense(nodes=400))
    model.add(Relu())
    model.add(Dense(nodes=10))
    model.add(Softmax())

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=200, lr = 1e-2, momentum=0.5, callbacks=callbacks)
    model.save("models/mnist_test_conv_2")

    mt.plot_training_progress()
    y_pred_prob = model.predict(x_train)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/check_gradients.py contains:
 #####################################################

import numpy as np
import sys, pathlib
from helper import read_mnist, read_cifar_10, read_names
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.utils import LoadXY
from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler

np.random.seed(1)

def evaluate_cost_W(W, x, y_real, l2_reg, filter_id):
    model.layers[0].filters[filter_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def evaluate_cost_b(W, x, y_real, l2_reg, bias_id):
    model.layers[0].biases[bias_id] = W
    y_pred = model.predict(x)
    c = model.cost(y_pred, y_real, l2_reg)
    return c

def ComputeGradsNum(x, y_real, model, l2_reg, h):
    """ Converted from matlab code """
    print("Computing numerical gradients...")

    grads_w = []
    for filter_id, filt in enumerate(model.layers[0].filters):
        W = copy.deepcopy(filt)  # Compute W
        grad_W = np.zeros(W.shape)
        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                for c in range(W.shape[2]):
                    W_try = np.array(W)
                    # print(W_try.shape)
                    W_try[i,j,c] -= h
                    c1 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    W_try = np.array(W)
                    W_try[i,j,c] += h
                    c2 = evaluate_cost_W(W_try, x, y_real, l2_reg, filter_id)
                    
                    grad_W[i,j,c] = (c2-c1) / (2*h)

                    model.layers[0].filters[filter_id] = W  # Reset it

        grads_w.append(grad_W)

    grads_b = []
    for bias_id, bias in enumerate(model.layers[0].biases):
        b_try = copy.deepcopy(bias) - h
        c1 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        b_try = copy.deepcopy(bias) + h
        c2 = evaluate_cost_b(b_try, x, y_real, l2_reg, bias_id)
        
        grad_b = (c2-c1) / (2*h)
        model.layers[0].biases[bias_id] = bias  # Reset it

        grads_b.append(grad_b)
    return grads_w, grads_b

if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=3, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=2, n_val=5, n_test=2)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_names(n_train=500)

    class_sum = np.sum(y_train, axis=1)*y_train.shape[0]
    class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)

    print(class_count)

    print(type(x_train[0, 0, 0]))

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=2, kernel_shape=(4, 4), stride=3, dilation_rate=2, input_shape=x_train.shape[0:-1]))
    model.add(Relu())
    model.add(MaxPool2D((2, 2), stride=3))
    model.add(Flatten())
    model.add(Dense(nodes=y_train.shape[0]))
    model.add(Relu())
    model.add(Softmax())

    print(np.min(np.abs(model.layers[0].filters)))

    reg = 0.0

    # Fit model
    anal_time = time.time()
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=200, epochs=1, lr=0, momentum=0, l2_reg=reg)
    analytical_grad_weight = model.layers[0].filter_gradients
    analytical_grad_bias = model.layers[0].bias_gradients
    # print(analytical_grad_weight)
    print(analytical_grad_bias)
    anal_time = time.time() - anal_time

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad_w, numerical_grad_b = ComputeGradsNum(x_train, y_train, model, l2_reg=reg, h=1e-5)
    # print(numerical_grad_w)
    print(numerical_grad_b)
    num_time = time.time() - num_time

    print("Weight Error:")
    _EPS = 0.0000001
    denom = np.abs(analytical_grad_weight) + np.abs(numerical_grad_w)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_weight-numerical_grad_w),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")


    print("Bias Error:")
    _EPS = 0.000000001
    denom = np.abs(analytical_grad_bias) + np.abs(numerical_grad_b)
    av_error = np.average(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    max_error = np.max(
            np.divide(
                np.abs(analytical_grad_bias-numerical_grad_b),
                np.multiply(denom, (denom > _EPS)) + np.multiply(_EPS*np.ones(denom.shape), (denom <= _EPS))))
    print("Averaged Element-Wise Relative Error:", av_error*100, "%")
    print("Max Element-Wise Relative Error:", max_error*100, "%")

    print("Speedup:", (num_time/anal_time))



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/cifar_10_test.py contains:
 #####################################################

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
    x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10(n_train=30000, n_val=200, n_test=200)
    # x_train, y_train, x_val, y_val, x_test, y_test = read_cifar_10()

    print(x_train.shape)
    # print(y_train.shape)

    # for i in range(200):
    #     cv2.imshow("image", x_train[..., i])
    #     cv2.waitKey()

    # Define callbacks
    mt = MetricTracker(file_name="cifar_test_3")  # Stores training evolution info (losses and metrics)
    # bms = BestModelSaver("models/best_cifar")  # Stores training evolution info (losses and metrics)
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
    # model.load("models/cifar_test_2")
    # mt.load("models/tracker")
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
              batch_size=100, epochs=20, momentum=0.9, l2_reg=0.003, callbacks=callbacks)
    model.save("models/cifar_test_3")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    # y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/names.py contains:
 #####################################################

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
    n1 = 40  # Filters of first Conv2D
    k1 = 6   # First kernel y size
    n2 = 20  # Filters of second Conv2D
    k2 = 4   # Second kernel y size
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
              batch_size=100, epochs=500, lr = 1e-3, momentum=0.8, l2_reg=0.001,
              compensate=True, callbacks=callbacks)
    model.save("models/names_best")

    mt.plot_training_progress(save=True, name="figures/names_best")
    # y_pred_prob = model.predict(x_train)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/confusion_matrices.py contains:
 #####################################################

import numpy as np
import sys, pathlib
from helper import read_names, read_names_countries
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import plot_confusion_matrix

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, _, _ = read_names(n_train=-1)
    print(x_train.shape)
    classes = read_names_countries()

    # Load model model
    model = Sequential(loss=CrossEntropy())
    # model.load("models/names_test")
    model.load("models/name_metaparam_search_2/n1-39_n2-33_k1-2_k2-10_batch_size-50")

    y_pred_train = model.predict_classes(x_train)
    y_pred_val = model.predict_classes(x_val)
    
    plot_confusion_matrix(y_pred_train, y_train, classes, "figures/conf_best_model")
    plot_confusion_matrix(y_pred_val, y_val, classes, "figures/conf_best_model_val")


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/metaparam_search.py contains:
 #####################################################
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
        "experiment_name": "name_metaparam_search_2",
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
                                      input_file="name_metaparam_search" + '/evaluations.csv',
                                      output_file=fixed_args["experiment_name"] + '/evaluations.csv')
    gp_search.init_session()
    x, y = gp_search.get_maximum(n_calls = 12,
                                 n_random_starts=0,
                                 noise=0.001,
                                 verbose=True)
    print("Max at:", x, "with value:", y)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/helper.py contains:
 #####################################################
import numpy as np

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
    
    filename = "data/" + filename
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data
    
def read_mnist(dim=[28,28],n_train=50000, n_val=10000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    
    import scipy.misc

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    def rs(imgs):
        imgs = (imgs.T).reshape((dim[0], dim[1], imgs.shape[0]), order='C')
        return np.expand_dims(imgs, axis=2).astype(float)  # h, w, c, n
    return rs(train_imgs[:n_train]),train_lbls_1hot[:n_train].T.astype(float),\
           rs(train_imgs[n_train:n_train+n_val]),train_lbls_1hot[n_train:n_train+n_val].T.astype(float),\
           rs(test_imgs[:n_test]),test_lbls_1hot[:n_test].T.astype(float)

def read_cifar_10(n_train=None, n_val=None, n_test=None):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
    from mlp.utils import LoadXY

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

    return x_train.astype(float), y_train.astype(float), x_val.astype(float), y_val.astype(float), x_test.astype(float), y_test.astype(float)

def read_names(n_train=-1):
    def read_file(filepath="data/names/ascii_names.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = -1
        for name in names:
            n_len = max(n_len, len(name))
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)

    val_indxs = []
    with open("data/names/Validation_Inds.txt") as fp:
        val_indxs = [int(val) for val in fp.readline().split(" ")]

    indx = list(range(len(names)))
    np.random.shuffle(indx)
    for val in val_indxs:
        indx.remove(val)

    return x[..., indx[:n_train]], y[..., indx[:n_train]],\
           x[..., val_indxs], y[..., val_indxs],\
           None, None


def read_names_test(n_train=-1):
    def read_file(filepath="data/names/test.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = 19
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)
    return x, y, names

def read_names_countries():
    return ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German",\
            "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese",\
            "Russian", "Scottish", "Spanish", "Vietnamese"]

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/test.py contains:
 #####################################################
import numpy as np
import sys, pathlib
import cv2

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.callbacks import MetricTracker

mt = MetricTracker()
mt.metric_name = "Accuracy"
mt.load("models/tracker")
mt.plot_training_progress()


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/mnist.py contains:
 #####################################################

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

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = read_mnist(n_train=200, n_val=200, n_test=2)

    # Define callbacks
    mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="linear", lr_min=1e-3, lr_max=9e-1)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define model
    model = Sequential(loss=CrossEntropy(), metric=Accuracy())
    model.add(Conv2D(num_filters=64, kernel_shape=(4, 4), input_shape=(28, 28, 1)))
    model.add(Relu())
    model.add(MaxPool2D(kernel_shape=(2, 2)))
    # model.add(Conv2D(num_filters=32, kernel_shape=(3, 3)))
    # model.add(Relu())
    model.add(Flatten())
    # model.add(Flatten(input_shape=(28, 28, 1)))
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
              batch_size=100, epochs=200, lr = 1e-2, momentum=0.5, callbacks=callbacks)
    model.save("models/mnist_test_conv_2")
    # model.layers[0].show_filters()

    # for filt in model.layers[0].filters:
    #     print(filt)

    # print(model.layers[0].biases)

    mt.plot_training_progress()
    y_pred_prob = model.predict(x_train)
    # # # model.pred
    # print(y_train)
    # print(np.round(y_pred_prob, decimals=2))


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_3/test_name_model.py contains:
 #####################################################


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/join_code.py contains:
 #####################################################
from glob import glob
import os

files = [f for f in glob('/home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/**', recursive=True) if os.path.isfile(f)]
files = [f for f in files if ".py" in f and ".pyc" not in f\
                                        and "/examples/" not in f\
                                        and "Assignment_1" not in f\
                                        and "Assignment_2" not in f]

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

try:  # If installed try to use parallelized einsum
    from einsum2 import einsum2 as einsum
except:
    print("Did not find einsum2, using numpy einsum (SLOWER)")
    from numpy import einsum as einsum


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
    def __init__(self, class_count=None):
        self._EPS = 1e-5
        self.classes_counts = class_count
        
    def __call__(self, Y_pred, Y_real):
        proportion_compensation = np.ones(Y_real.shape[-1])
        if self.classes_counts is not None:
            proportion_compensation = np.dot(Y_real.T, self.classes_counts)

        logs = np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0))
        prod = np.dot(logs, proportion_compensation)
        return -prod/float(Y_pred.shape[1])

    def backward(self, Y_pred, Y_real):
        proportion_compensation = np.ones(Y_real.shape[-1])
        if self.classes_counts is not None:
            proportion_compensation = np.dot(Y_real.T, self.classes_counts)
        # d(-log(x))/dx = -1/x
        f_y = np.multiply(Y_real, Y_pred)
        # Element-wise inverse
        loss_diff = - \
            np.reciprocal(f_y, out=np.zeros_like(
                Y_pred), where=abs(f_y) > self._EPS)
        # Account for class imbalance
        loss_diff = loss_diff*proportion_compensation
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
import time

try:  # If installed try to use parallelized einsum
    from einsum2 import einsum2 as einsum
except:
    print("Did not find einsum2, using numpy einsum (SLOWER)")
    from numpy import einsum as einsum

# Layer Templates #######################################################


class Layer(ABC):
    """ Abstact class to represent Layers
        A Layer has:
            - compile  method which computes input/output shapes (and intializes weights)
            - __call__ method to apply Layer function to the inputs
            - gradient method to add Layer function gradient to the backprop
    """
    name = "GenericLayer"

    def __init__(self):
        self.weights = None
        self.is_compiled = False
        self.input_shape = None
        self.output_shape = None
        # NOTE: INPUT/OUTPUT shapes ignore

    @abstractmethod
    def compile(self, input_shape):
        """ Updates self.input_shape (and self.output_shape)
            If input_dim not set by user, when added to a model 
            this method is called based on previous layer output_shape

            For trainable layers, it also should initializes weights and
            gradient placeholders according to input_shape.

            Note: input_shape should be a tuple of the shape
            ignoring the number of samples (last elem)
        """
        input_shape = input_shape if type(
            input_shape) is tuple else (input_shape, )
        self.input_shape = input_shape
        self.is_compiled = True
        # print("compiled")

    @abstractmethod
    def __call__(self, inputs):
        """ Applies a function to given inputs """
        pass

    @abstractmethod
    def backward(self, in_gradient):
        """ Receives right-layer gradient and multiplies it by current layer gradient """
        pass


class ConstantShapeLayer(Layer):
    """ Common structure of Layers which do not modify the shape of input and output
    """

    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is not None:
            self.compile(input_shape)

    def compile(self, input_shape):
        super().compile(input_shape)  # Populates self.input_shape and self.is_compiled
        self.output_shape = self.input_shape


# Activation Layers #####################################################
class Softmax(ConstantShapeLayer):
    def __call__(self, x):
        self.outputs = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient


class Relu(ConstantShapeLayer):
    def __call__(self, x):
        self.inputs = x
        return np.multiply(x, (x > 0))

    def backward(self, in_gradient, **kwargs):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

# MISC LAYERS ###########################################################


class Dropout(ConstantShapeLayer):
    def __init__(self, ones_ratio=0.7):
        self.name = "Dropout"
        self.ones_ratio = ones_ratio

    def __call__(self, x, apply=True):
        if apply:
            self.mask = np.random.choice([0, 1], size=(x.shape), p=[
                                         1 - self.ones_ratio, self.ones_ratio])
            return np.multiply(self.mask, x)
        return x

    def backward(self, in_gradient, **kwargs):
        return np.multiply(self.mask, in_gradient)


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        if input_shape is not None:
            self.compile(input_shape)

    def compile(self, input_shape):
        super().compile(input_shape)  # Updates self.input_shape and self.is_compiled
        self.output_shape = (np.prod(self.input_shape),)

    def __call__(self, inputs):
        self.__in_shape = inputs.shape  # Store inputs shape to use in backprop
        m = self.output_shape[0]
        n_points = inputs.shape[3]
        return inputs.reshape((m, n_points))

    def backward(self, in_gradient, **kwargs):
        return np.array(in_gradient).reshape(self.__in_shape)


class MaxPool2D(Layer):
    def __init__(self, kernel_shape=(2, 2), stride=1, input_shape=None):
        super().__init__()
        self.kernel_shape = kernel_shape
        self.s = stride  # TODO(oleguer) Implement stride
        if input_shape is not None:
            self.compile(input_shape)  # Only care about channels

    def compile(self, input_shape):
        # Input shape must be (height, width, channels,)
        assert(len(input_shape) == 3)
        super().compile(input_shape)
        (ker_h, ker_w) = self.kernel_shape
        out_h = int((input_shape[0] - ker_h)/self.s) + 1
        out_w = int((input_shape[1] - ker_w)/self.s) + 1
        self.output_shape = (out_h, out_w, input_shape[2],)

    def __call__(self, inputs):
        """ Forward pass of MaxPool2D
            input should have shape (height, width, channels, n_images)
        """
        assert(len(inputs.shape) ==
               4)  # Input must have shape (height, width, channels, n_images)
        # Set input shape does not match with input sent
        assert(inputs.shape[:3] == self.input_shape)

        # Get shapes
        (ker_h, ker_w) = self.kernel_shape
        (out_h, out_w, _,) = self.output_shape

        # Compute convolution
        self.inputs = inputs  # Will be used in back pass
        output = np.empty(shape=self.output_shape + (self.inputs.shape[3],))
        for i in range(out_h):
            for j in range(out_w):
                # TODO(oleguer): Not sure if np.amax is parallel, look into  numexpr
                in_block = self.inputs[self.s*i:self.s *
                                       i+ker_h, self.s*j:self.s*j+ker_w, :, :]
                output[i, j, :, :] = np.amax(in_block, axis=(0, 1,))
        return output

    def backward(self, in_gradient, **kwargs):
        """ Pass gradient to left layer """
        # Get shapes
        (out_h, out_w, n_channels, n_points) = in_gradient.shape
        (ker_h, ker_w) = self.kernel_shape

        # Incoming gradient shape must match layer output shape
        assert(out_h == self.output_shape[0])
        # Incoming gradient shape must match layer output shape
        assert(out_w == self.output_shape[1])

        # Instantiate gradients
        left_layer_gradient = np.zeros(
            self.input_shape + (in_gradient.shape[-1],))
        for i in range(out_h):
            for j in range(out_w):
                in_block = self.inputs[self.s*i:self.s *
                                       i+ker_h, self.s*j:self.s*j+ker_w, :, :]
                mask = np.equal(in_block, np.amax(
                    in_block, axis=(0, 1,))).astype(int)
                masked_gradient = mask*in_gradient[i, j, :, :]
                left_layer_gradient[self.s*i:self.s*i+ker_h,
                                    self.s*j:self.s*j+ker_w, :, :] += masked_gradient
        return left_layer_gradient

# TRAINABLE LAYERS ######################################################


class Dense(Layer):
    def __init__(self, nodes, input_dim=None, weight_initialization="in_dim"):
        super().__init__()
        self.nodes = nodes
        self.weight_initialization = weight_initialization
        if input_dim is not None:  # If user sets input, automatically compile
            self.compile(input_dim)

    def compile(self, input_shape):
        super().compile(input_shape)  # Updates self.input_shape and self.is_compiled
        self.__initialize_weights(self.weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates
        self.output_shape = (self.nodes,)

    def __call__(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return np.dot(self.weights, self.inputs)

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        # Remove bias TODO Think about this
        left_layer_gradient = (np.dot(self.weights.T, in_gradient))[:-1, :]

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        # TODO: Rremove self if not going to update it
        self.gradient = np.dot(
            in_gradient, self.inputs.T) + regularization_term
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "normal":
            self.weights = np.array(np.random.normal(
                0.0, 1./100.,
                                    (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.array(np.random.normal(
                0.0, 1./float(np.sqrt(self.input_shape[0])),
                (self.nodes, self.input_shape[0]+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape[0]))
            self.weights = np.array(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape[0]+1)))  # Add biases


class Conv2D(Layer):
    def __init__(self, num_filters=5, kernel_shape=(5, 5), stride=1, dilation_rate=1, input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.s = stride
        self.dilation = dilation_rate
        self.kernel_shape = kernel_shape
        aug_ker_h = (self.kernel_shape[0]-1)*self.dilation + 1
        aug_ker_w = (self.kernel_shape[1]-1)*self.dilation + 1
        # Kernel size considering dilation_rate
        self.aug_kernel_shape = (aug_ker_h, aug_ker_w)
        if input_shape is not None:
            self.compile(input_shape)  # Only care about channels

    def compile(self, input_shape):
        # Input shape must be (height, width, channels,)
        assert(len(input_shape) == 3)
        super().compile(input_shape)
        (ker_h, ker_w) = self.aug_kernel_shape
        out_h = int((input_shape[0] - ker_h)/self.s) + 1
        out_w = int((input_shape[1] - ker_w)/self.s) + 1
        self.output_shape = (out_h, out_w, self.num_filters,)
        self.__initialize_weights()

    def __call__(self, inputs):
        """ Forward pass of Conv Layer
            input should have shape (height, width, channels, n_images)
            channels should match kernel_shape
        """
        assert(len(inputs.shape) == 4)  # Input (height, width, channels, n_images)
        # Set input shape does not match with input sent
        assert(inputs.shape[:3] == self.input_shape)
        # Filter number of channels must match input channels
        assert(self.filters.shape[3] == inputs.shape[2])

        # Get shapes
        (aug_ker_h, aug_ker_w) = self.aug_kernel_shape
        (out_h, out_w, _,) = self.output_shape

        # Compute convolution
        self.inputs = inputs  # Will be used in back pass
        output = np.empty(shape=self.output_shape + (self.inputs.shape[3],))
        for i in range(out_h):
            for j in range(out_w):
                in_block = inputs[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                  self.s*j:self.s*j+aug_ker_w:self.dilation, :, :]
                output[i, j, :, :] = einsum(
                    "ijcn,kijc->kn", in_block, self.filters)

        # Add biases
        output += einsum("ijcn,c->ijcn", np.ones(output.shape), self.biases)
        return output

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        """ Weight update
        """
        # Get shapes
        (out_h, out_w, _, _) = in_gradient.shape
        (aug_ker_h, aug_ker_w) = self.aug_kernel_shape

        # Incoming gradient shape must match layer output shape
        assert(out_h == self.output_shape[0])
        # Incoming gradient shape must match layer output shape
        assert(out_w == self.output_shape[1])

        # Instantiate gradients
        left_layer_gradient = np.zeros(
            self.input_shape + (in_gradient.shape[-1],))
        # Save it to compare with numerical (DEBUG)
        self.filter_gradients = np.zeros(self.filters.shape)
        self.bias_gradients = np.sum(in_gradient, axis=(0, 1, 3))

        for i in range(out_h):
            for j in range(out_w):
                in_block = self.inputs[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                       self.s*j:self.s*j+aug_ker_w:self.dilation, :, :]
                grad_block = in_gradient[i, j, :, :]
                filter_grad = einsum("ijcn,kn->kijc", in_block, grad_block)
                self.filter_gradients += filter_grad
                left_layer_gradient[self.s*i:self.s*i+aug_ker_h:self.dilation,
                                    self.s*j:self.s*j+aug_ker_w:self.dilation, :, :] +=\
                    einsum("kijc,kn->ijcn", self.filters, grad_block)


        self.filter_gradients += 2*l2_regularization*self.filters

        if np.array_equal(self.dw, np.zeros(self.filters.shape)):
            self.dw = self.filter_gradients
        else:
            self.dw = momentum*self.dw + (1-momentum)*self.filter_gradients

        self.filters -= lr*self.dw  # TODO(oleguer): Add regularization
        self.biases -= lr*self.bias_gradients
        return left_layer_gradient

    def __initialize_weights(self):
        self.filters = []
        self.biases = np.random.normal(0.0, 1./100., self.num_filters)
        # self.biases = np.zeros(self.num_filters)
        full_kernel_shape = self.kernel_shape + (self.input_shape[2],)
        # self.biases = np.array([0, 1])
        for i in range(self.num_filters):
            kernel = np.random.normal(0.0, 1./100., full_kernel_shape)
            # kernel = np.ones(full_kernel_shape)/3
            if len(kernel.shape) == 2:
                kernel = np.expand_dims(kernel, axis=2)
            self.filters.append(kernel)
        self.filters = np.array(self.filters)
        self.dw = np.zeros(self.filters.shape)

    def show_filters(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(self.filters.shape[0])
        for i in range(self.filters.shape[0]):
            axes[i].imshow(self.filters[i][:, :, 0])
        plt.show()


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
        if len(self.layers) > 1: # Compile layer using output shape of previous layer
            assert(self.layers[-2].is_compiled)  # Input/Output shapes not set for previous layer!
            # Set input shape to be previous layer output_shape
            self.layers[-1].compile(input_shape=self.layers[-2].output_shape)
        print(layer.input_shape)
        # print(layer.output_shape)

    def predict(self, X, apply_dropout=True):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            if layer.name == "Dropout":
                vals = layer(vals, apply_dropout)
            else:
                vals = layer(vals)
        return vals

    def predict_classes(self, X):
        Y_pred_prob = self.predict(X, apply_dropout=False)
        idx = np.argmax(Y_pred_prob, axis=0)
        Y_pred_class = np.zeros(Y_pred_prob.shape)
        Y_pred_class[idx, np.arange(Y_pred_class.shape[1])] = 1
        return Y_pred_class

    def get_metric_loss(self, X, Y_real, use_dropout=True):
        """ Returns loss and value of success metric
        """
        if X is None or Y_real is None:
            print("Attempting to get metrics of None")
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
            shuffle_minibatch=True, compensate=False, callbacks=[], **kwargs):
        """ Performs backrpop with given parameters.
            save_path is where model of best val accuracy will be saved
        """
        assert(epochs is None or iterations is None) # Only one can set it limit
        if iterations is not None:
            epochs = int(np.ceil(iterations/(X.shape[-1]/batch_size)))
        else:
            iterations = int(epochs*np.ceil((X.shape[-1]/batch_size)))

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
        self.train_metric = 0
        self.val_metric = 0
        self.train_loss = 0
        self.val_loss = 0
        self.t = 0

        # Call callbacks
        for callback in callbacks:
            callback.on_training_begin(self)

        # Training
        stop = False
        pbar = tqdm(list(range(self.epochs)))
        for self.epoch in pbar:
        # for self.epoch in range(self.epochs):
            for X_minibatch, Y_minibatch in minibatch_split(X, Y, batch_size, shuffle_minibatch, compensate):
                # t = time.time()
                self.Y_pred_prob = self.predict(X_minibatch)  # Forward pass
                # print("forward_time:", time.time()-t)
                # t = time.time()
                gradient = self.loss.backward(
                    self.Y_pred_prob, Y_minibatch)  # Loss grad
                # print("loss_backward_time:", time.time()-t)
                # print("backward")
                for layer in reversed(self.layers):  # Backprop (chain rule)
                    # t = time.time()
                    gradient = layer.backward(
                        in_gradient=gradient,
                        lr=self.lr,  # Trainable layer parameters
                        momentum=self.momentum,
                        l2_regularization=self.l2_reg)
                    # print("gradient_time:", layer, time.time()-t)
                # t = time.time()
                # Call callbacks
                for callback in callbacks:
                    callback.on_batch_end(self)
                if self.t >= iterations:
                    stop = True
                    break
                self.t += 1  # Step counter
                # print("batch_callbacks_time:", layer, time.time()-t)
            # t = time.time()
            # Call callbacks
            for callback in callbacks:
                callback.on_epoch_end(self)
            # print("epoch_callbacks_time:", layer, time.time()-t)

            # Update progressbar
            pbar.set_description("Train acc: " + str(np.round(self.train_metric*100, 2)) +\
                                 "% Val acc: " + str(np.round(self.val_metric*100, 2)) +\
                                 "% Train Loss: " + str(np.round(self.train_loss)))
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

    def __init__(self, file_name="models/tracker"):
        self.file_name = file_name
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []

    def on_training_begin(self, model):
        self.metric_name = model.metric.name
        # self.__track(model)

    def on_batch_end(self, model):
        # self.__track(model)
        pass

    def on_epoch_end(self, model):
        self.__track(model)
        self.save(self.file_name)
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
        model.train_metric = train_metric
        model.train_loss = train_loss

    def plot_training_progress(self, show=True, save=False, name="model_results", subtitle=None):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(bottom=0)
        ax1.set_ylim(top=1.25*np.nanmax(self.val_losses))
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='upper left')

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
        # np.save(file + "_lr", self.learning_rates)
        np.save(file + "_train_met", self.train_metrics)
        np.save(file + "_val_met", self.val_metrics)
        np.save(file + "_train_loss", self.train_losses)
        np.save(file + "_val_loss", self.val_losses)

    def load(self, file):
        self.train_metrics = np.load(file + "_train_met.npy").tolist()
        self.val_metrics = np.load(file + "_val_met.npy").tolist()
        self.train_losses = np.load(file + "_train_loss.npy").tolist()
        self.val_losses = np.load(file + "_val_loss.npy").tolist()


class BestModelSaver(Callback):
    def __init__(self, save_dir=None):
        self.save_dir = None
        if save_dir is not None:
            self.save_dir = os.path.join(save_dir, "best_model")
        self.best_metric = -np.inf
        self.best_model_layers = None
        self.best_model_loss = None
        self.best_model_metric = None

    def on_epoch_end(self, model):
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
	return np.array(dataset[b"data"]).T, np.array(one_hot_labels).T

def LoadXY(filename):
	return getXY(LoadBatch(filename))

def plot(flatted_image, shape=(32, 32, 3), order='F'):
	image = np.reshape(flatted_image, shape, order=order)
	cv2.imshow("image", image)
	cv2.waitKey()

def accuracy(Y_pred_classes, Y_real):
	return np.sum(np.multiply(Y_pred_classes, Y_real))/Y_pred_classes.shape[1]

def minibatch_split(X, Y, batch_size, shuffle=True, compansate=False):
	"""Yields splited X, Y matrices in minibatches of given batch_size"""
	if (batch_size is None) or (batch_size > X.shape[-1]):
		batch_size = X.shape[-1]

	if not compansate:
		indx = list(range(X.shape[-1]))
		if shuffle:
			np.random.shuffle(indx)
		for i in range(int(X.shape[-1]/batch_size)):
			pos = i*batch_size
			# Get minibatch
			X_minibatch = X[..., indx[pos:pos+batch_size]]
			Y_minibatch = Y[..., indx[pos:pos+batch_size]]
			if i == int(X.shape[-1]/batch_size) - 1:  # Get all the remaining
				X_minibatch = X[..., indx[pos:]]
				Y_minibatch = Y[..., indx[pos:]]
			yield X_minibatch, Y_minibatch
	else:
		class_sum = np.sum(Y, axis=1)*Y.shape[0]
		class_count = np.reciprocal(class_sum, where=abs(class_sum) > 0)
		x_probas = np.dot(class_count, Y)
		n = X.shape[-1]
		for i in range(int(n/batch_size)):
			indxs = np.random.choice(range(n), size=batch_size, replace=True, p=x_probas)
			yield X[..., indxs], Y[..., indxs]

def plot_confusion_matrix(Y_pred, Y_real, class_names, path=None):
	heatmap = np.zeros((Y_pred.shape[0], Y_pred.shape[0]))
	for n in range(Y_pred.shape[-1]):
		i = np.where(Y_pred[:, n]==1)[0][0]
		j = np.where(Y_real[:, n]==1)[0][0]
		heatmap[i, j] += 1
		
	import seaborn as sn
	import pandas as pd
	
	df_cm = pd.DataFrame(heatmap, index = [i for i in class_names],
								  columns = [i for i in class_names])
	plt.figure(figsize = (10,10))
	sn.heatmap(df_cm, robust=True, square=True,annot=True, fmt='g')
	if path is not None:
		plt.savefig(path)
	plt.show()

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

