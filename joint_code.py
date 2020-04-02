#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/joint_code.py contains:
 #####################################################

from glob import glob
import os

files = [f for f in glob('/home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/**', recursive=True) if os.path.isfile(f)]
files = [f for f in files if ".py" in f and ".pyc" not in f and "/examples/" not in f]

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
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/layers.py contains:
 #####################################################
import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
import sys


class Dense:
    def __init__(self, nodes, input_dim, weight_initialization="fixed"):
        self.nodes = nodes
        self.input_shape = input_dim
        self.__initialize_weights(weight_initialization)
        self.dw = np.zeros(self.weights.shape)  # Weight updates

    def forward(self, inputs):
        self.inputs = np.append(
            inputs, [np.ones(inputs.shape[1])], axis=0)  # Add biases
        return self.weights*self.inputs

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        # Previous layer error propagation
        left_layer_gradient = (
            self.weights.T*in_gradient)[:-1, :]  # Remove bias TODO Think about this

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Bias col to 0
        regularization_term = 2*l2_regularization * \
            regularization_weights  # Only current layer weights != 0

        # Weight update
        self.gradient = in_gradient*self.inputs.T + regularization_term #TODO: Rremove self if not going to update it
        self.dw = momentum*self.dw + (1-momentum)*self.gradient
        self.weights -= lr*self.dw
        # print(np.average(np.abs(self.dw)))
        # print(np.average(np.abs(self.dw[:, -1])))
        # print("#####")
        return left_layer_gradient

    def __initialize_weights(self, weight_initialization):
        if weight_initialization == "fixed":
            self.weights = np.matrix(np.random.normal(
                0, 1./100.,
                                    (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "in_dim":
            self.weights = np.matrix(np.random.normal(
                0, 1./float(self.input_shape),
                (self.nodes, self.input_shape+1)))  # Add biases
        if weight_initialization == "xavier":
            limit = np.sqrt(6/(self.nodes+self.input_shape))
            self.weights = np.matrix(np.random.uniform(
                low=-limit,
                high=limit,
                size=(self.nodes, self.input_shape+1)))  # Add biases


class Activation:
    def __init__(self, activation="softmax"):
        self.activation_type = activation
        self.weights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = None
        if self.activation_type == "softmax":
            self.outputs = self.__softmax(x=inputs)
        if self.activation_type == "relu":
            self.outputs = self.__relu(x=inputs)
        return self.outputs

    def backward(self, in_gradient, **kwargs):
        # Gradient of activation function evaluated at pre-activation
        if self.activation_type == "softmax":
            return self.__softmax_diff(in_gradient=in_gradient)
        if self.activation_type == "relu":
            return self.__relu_diff(in_gradient=in_gradient)
        return None

    # ACTIVATION FUNCTIONS ###########################################################
    # TODO(oleguer): Maybe this shouldt be here
    def __relu(self, x):
        return np.multiply(x, (x > 0))

    def __relu_diff(self, in_gradient):
        # TODO(Oleguer): review this
        return np.multiply((self.inputs > 0), in_gradient)

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def __softmax_diff(self, in_gradient):
        diags = np.einsum("ik,ij->ijk", self.outputs,
                          np.eye(self.outputs.shape[0]))
        out_prod = np.einsum("ik,jk->ijk", self.outputs, self.outputs)
        gradient = np.einsum("ijk,jk->ik", (diags - out_prod), in_gradient)
        return gradient


if __name__ == "__main__":
    pass


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Toy-DeepLearning-Framework/mlp/models.py contains:
 #####################################################
import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import pickle

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from layers import Activation, Dense
from utils import prob_to_class, accuracy, minibatch_split

class Sequential:
    def __init__(self, loss="cross_entropy", pre_saved=None):
        self.layers = []
        self.loss_type = loss
        if pre_saved is not None:
            self.load(pre_saved)

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)

    def predict(self, X):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            vals = layer.forward(vals)
        return vals

    def get_classification_metrics(self, X, Y_real):
        """ Returns loss and classification accuracy """
        if X is None or Y_real is None:
            return 1, 0
        Y_pred_prob = self.predict(X)
        Y_pred_classes = prob_to_class(Y_pred_prob)
        acc = accuracy(Y_pred_classes, Y_real)
        loss = self.__loss(Y_pred_prob, Y_real)
        return acc, loss

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=100, lr=0.01, momentum=0.7, l2_reg=0.1, save_path=None):
        """ Performs backrpop with given parameters.
            save_path is where model of best val accuracy will be saved
        """
        # Restart tracking the learning
        best_model = None
        max_val_acc = self.__track_training(X, Y, X_val, Y_val, restart=True)
        # Training
        pbar = tqdm(list(range(epochs)))
        for epoch in pbar:
            for X_minibatch, Y_minibatch in minibatch_split(X, Y, batch_size):
                Y_pred_prob = self.predict(X_minibatch)  # Forward pass
                gradient = self.__loss_diff(Y_pred_prob, Y_minibatch)  # Loss grad
                for layer in reversed(self.layers):  # Backprop (chain rule)
                    gradient = layer.backward(
                        in_gradient=gradient,
                        lr=lr,  # Trainable layer parameters
                        momentum=momentum,
                        l2_regularization=l2_reg)
            # TODO(Oleguer): ALL THOSE SHOULD BE CALLBACKS PASSED BY USER!!!
            val_acc = self.__track_training(X, Y, X_val, Y_val)  # Update tracking
            if save_path is not None and val_acc > max_val_acc:  # Save model if improved val_Acc
                max_val_acc = val_acc
                self.save(save_path)
                best_model = copy.deepcopy(self)  # TODO(oleguer): Probably there is a decent way of doing this
            pbar.set_description("Val acc: " + str(val_acc))
            lr = 0.9*lr  # Weight decay TODO(oleguer): Do this in a scheduler

        # # Set latest tracking TODO(oleguer) Use a dictionary or something!!
        best_model.train_accuracies = self.train_accuracies
        best_model.val_accuracies = self.val_accuracies
        best_model.train_losses = self.train_losses
        best_model.val_losses = self.val_losses
        return best_model
    
    def plot_training_progress(self, show=True, save=False, name="model_results", subtitle=None):
        fig, ax1 = plt.subplots()
        # Losses
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim(bottom=np.amin(self.val_losses)/2)
        ax1.set_ylim(top=1.25*np.amax(self.val_losses))
        if len(self.val_losses) > 0:
            ax1.plot(list(range(len(self.val_losses))),
                     self.val_losses, label="Val loss", c="red")
        ax1.plot(list(range(len(self.train_losses))),
                 self.train_losses, label="Train loss", c="orange")
        ax1.tick_params(axis='y')
        plt.legend(loc='center right')

        # Accuracies
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(bottom=0)
        ax2.set_ylim(top=0.5)
        n = len(self.train_accuracies)
        ax2.plot(list(range(n)),
                 np.array(self.train_accuracies), label="Train acc", c="green")
        if len(self.val_accuracies) > 0:
            n = len(self.val_accuracies)
            ax2.plot(list(range(n)),
                     np.array(self.val_accuracies), label="Val acc", c="blue")
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

    def cost(self, Y_pred_prob, Y_real, l2_reg):
        """Computes cost = loss + regularization"""
        # Loss
        loss = 0
        if self.loss_type == "cross_entropy":
            loss = self.__cross_entropy(Y_pred_prob, Y_real)
        elif self.loss_type == "categorical_hinge":
            loss = self.__categorical_hinge(Y_pred_prob, Y_real)

        # Regularization
        w_norm = 0
        for layer in self.layers:
            if layer.weights is not None:
                w_norm += np.linalg.norm(layer.weights, 'fro')**2

        return loss + l2_reg*w_norm

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

    # Private methods
    def __track_training(self, X, Y, X_val=None, Y_val=None, restart=False):
        if restart:
            self.train_accuracies = []
            self.val_accuracies = []
            self.train_losses = []
            self.val_losses = []
        # TODO(oleguer): Allow for other metrics
        train_acc, train_loss = self.get_classification_metrics(X, Y)
        val_acc, val_loss = self.get_classification_metrics(X_val, Y_val)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        return val_acc

    # LOSS FUNCTIONS ##############################################
    # TODO(Oleguer): Should all this be here?

    def __loss(self, Y_pred_prob, Y_real):
        if self.loss_type == "cross_entropy":
            return self.__cross_entropy(Y_pred_prob, Y_real)
        if self.loss_type == "categorical_hinge":
            return self.__categorical_hinge(Y_pred_prob, Y_real)
        return None

    def __loss_diff(self, Y_pred, Y_real):
        if self.loss_type == "cross_entropy":
            return self.__cross_entropy_diff(Y_pred, Y_real)
        if self.loss_type == "categorical_hinge":
            return self.__categorical_hinge_diff(Y_pred, Y_real)
        return None

    def __cross_entropy(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=0)))/float(Y_pred.shape[1])

    def __cross_entropy_diff(self, Y_pred, Y_real):
        _EPS = 1e-5
        # d(-log(x))/dx = -1/x
        f_y = np.multiply(Y_real, Y_pred)
        # Element-wise inverse
        loss_diff = - \
            np.reciprocal(f_y, out=np.zeros_like(
                Y_pred), where=abs(f_y) > _EPS)
        return loss_diff/float(Y_pred.shape[1])

    def __categorical_hinge(self, Y_pred, Y_real):
        # L = SUM_data (SUM_dim_j(not yi) (MAX(0, y_pred_j - y_pred_yi + 1)))
        pos = np.sum(np.multiply(Y_real, Y_pred), axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        val = neg + 1. - pos
        val = np.multiply(val, (val > 0))
        return np.sum(val)/float(Y_pred.shape[1])

    def __categorical_hinge_diff(self, Y_pred, Y_real):
        # Forall j != yi: (y_pred_j - y_pred_yi + 1 > 0)
        # If     j == yi: -1 SUM_j(not yi) (y_pred_j - y_pred_yi + 1 > 0)
        pos = np.sum(np.multiply(Y_real, Y_pred), axis=0)  # Val of right result
        neg = np.multiply(1-Y_real, Y_pred)  # Val of wrong results
        # print((neg + 1. - pos > 0))
        # print(1-Y_real)
        wrong_class_activations = np.multiply(1-Y_real, (neg + 1. - pos > 0))  # Val of wrong results
        wca_sum = np.sum(wrong_class_activations, axis=0)
        # print(wca_sum)
        neg_wca = np.einsum("ij,j->ij", Y_real, np.array(wca_sum).flatten())
        return (wrong_class_activations - neg_wca)/float(Y_pred.shape[1])


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

def plot(flatted_image, shape=(32, 32, 3), order='F'):
	image = np.reshape(flatted_image, shape, order=order)
	cv2.imshow("image", image)
	cv2.waitKey()

def prob_to_class(Y_pred_prob):
	"""Given array of prob, returns max prob in one-hot fashon"""
	idx = np.argmax(Y_pred_prob, axis=0)
	Y_pred_class = np.zeros(Y_pred_prob.shape)
	Y_pred_class[idx, np.arange(Y_pred_class.shape[1])] = 1
	return Y_pred_class

def accuracy(Y_pred_classes, Y_real):
	return np.sum(np.multiply(Y_pred_classes, Y_real))/Y_pred_classes.shape[1]

def minibatch_split(X, Y, batch_size):
	"""Yields splited X, Y matrices in minibatches of given batch_size"""
	if (batch_size is None) or (batch_size > X.shape[1]):
		batch_size = X.shape[1]
	indx = list(range(X.shape[1]))
	np.random.shuffle(indx)
	for i in range(int(X.shape[1]/batch_size)):
		# Get minibatch
		X_minibatch = X[:, indx[i:i+batch_size]]
		Y_minibatch = Y[:, indx[i:i+batch_size]]
		if i == int(X.shape[1]/batch_size) - 1:  # Get all the remaining
			X_minibatch = X[:, indx[i:]]
			Y_minibatch = Y[:, indx[i:]]
		yield X_minibatch, Y_minibatch

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

def g(a, b, c):
    print(a)
    print(b)
    print(c)

def f(a, **kwargs):
    print(a)
    g(a, **kwargs)
    s = dict_to_string(kwargs)
    print(s)

if __name__ == "__main__":
    param_space = {
        "a": 1,
        "b": 2,
        "c": 3
    }
    f(**param_space)



#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/check_gradients_svm.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time
from tqdm import tqdm

from mlp.models import Sequential
from mlp.layers import Activation, Dense
from mlp.utils import getXY, LoadBatch, prob_to_class

np.random.seed(0)

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
            W_try = copy.deepcopy(W)
            W_try[i,j] -= h
            c1 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            W_try = copy.deepcopy(W)
            W_try[i,j] += h
            c2 = evaluate_cost(W_try, x, y_real, l2_reg)
            
            grad_W[i,j] = (c2-c1) / (2*h)
    return grad_W

if __name__ == "__main__":
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    # x_val, y_val = getXY(LoadBatch("data_batch_2"))
    # x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    # x_val = (x_val - mean_x)/std_x
    # x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0:5]
    y = y_train[:, 0:5]
    reg = 0.1

    # Define model
    model = Sequential(loss="categorical_hinge")
    model.add(Dense(nodes=10, input_dim=x.shape[0], weight_initialization="fixed"))

    anal_time = time.time()
    model.fit(x, y,
              batch_size=10000, epochs=1, lr=0, # 0 lr will not change weights
              momentum=0, l2_reg=reg)
    analytical_grad = model.layers[0].gradient
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, l2_reg=reg, h=0.001)
    print(numerical_grad.shape)
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
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/param_testing.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
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

def montage(W, title, path=None):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    plt.suptitle(title)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    if path is not None:
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()
    

# Define evaluator (function to run in MetaParamOptimizer)
def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_path="", **kwargs):
    # Define model
    model = Sequential(loss="cross_entropy")
    model.add(
        Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    model.add(Activation("softmax"))

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val, **kwargs)
    test_acc = model.get_classification_metrics(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) + ", Test Acc: " + str(test_acc)
    model.plot_training_progress(show=False,
                                save=True,
                                name="figures/param_testing/" + dict_to_string(kwargs),
                                subtitle=subtitle)
    model.save(experiment_path + "/" + dict_to_string(kwargs))
    montage(W=np.array(model.layers[0].weights[:, :-1]),
            title=subtitle,
            path="figures/param_testing/" + dict_to_string(kwargs) + "_weights")

    # Minimizing value: validation accuracy
    val_acc = model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": model}  # Save score and model
    return result

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

    # Load data
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    x_val, y_val = getXY(LoadBatch("data_batch_2"))
    x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Define list of parameters to try
    dicts_list = [
        { "l2_reg": 0.0, "lr": 0.1 },
        { "l2_reg": 0.0, "lr": 0.001 },
        { "l2_reg": 0.1, "lr": 0.001 },
        { "l2_reg": 1.0, "lr": 0.001 },
    ]
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_path" : "models/param_testing/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "x_test" : x_test,
        "y_test" : y_test,
        "batch_size": 100,
        "epochs" : 40,
        "momentum" : 0.,
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path=fixed_args["experiment_path"])
    best_model = mpo.list_search(evaluator=evaluator,
                                dicts_list=dicts_list,
                                fixed_args=fixed_args)

    model = Sequential(loss="cross_entropy")
    model.add(
        Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    model.add(Activation("softmax"))
    # model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
    #         batch_size=500, epochs=2, lr=0.001, # 0 lr will not change weights
    #         momentum=0, l2_reg=0.1)
    # montage(W=np.array(model.layers[0].weights[:, :-1]), title="bla bla")

    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/check_gradients.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import copy
import time

from mlp.models import Sequential
from mlp.layers import Activation, Dense
from mlp.utils import getXY, LoadBatch, prob_to_class

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
    for i in range(W.shape[0]):
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
    x_train, y_train = getXY(LoadBatch("data_batch_1"))
    x_val, y_val = getXY(LoadBatch("data_batch_2"))
    x_test, y_test = getXY(LoadBatch("test_batch"))

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    x = x_train[:, 0]
    y = y_train[:, 0]
    reg = 0.1

    # Define model
    model = Sequential(loss="cross_entropy")
    model.add(Dense(nodes=10, input_dim=x.shape[0], weight_initialization="fixed"))
    model.add(Activation("softmax"))

    anal_time = time.time()
    model.fit(x, y, batch_size=10000, epochs=1, lr=0, # 0 lr will not change weights
                    momentum=0, l2_reg=reg)
    analytical_grad = model.layers[0].gradient
    anal_time = anal_time - time.time()

    # Get Numerical gradient
    num_time = time.time()
    numerical_grad = ComputeGradsNum(x, y, model, l2_reg=reg, h=0.001)
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
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/svm.py contains:
 #####################################################
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
                        momentum=0.5, l2_reg=0.05, save_path="models/svm/test_2")
    best_model.plot_training_progress(show=False,
                                save=True,
                                name="figures/svm/test_2",
                                subtitle="subtitle")
    test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    val_acc = best_model.get_classification_metrics(x_val, y_val)[0]
    print("test_acc:", test_acc)
    print("val_acc:", val_acc)

#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/performance_optimization_svm.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
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

def montage(W, title, path=None):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    plt.suptitle(title)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    if path is not None:
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()
    

# Define evaluator (function to run in MetaParamOptimizer)
def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", init="fixed", **kwargs):
    # Define model
    model = Sequential(loss="categorical_hinge")
    model.add(Dense(nodes=10, input_dim=x.shape[0], weight_initialization="fixed"))

    # Fit model
    model_save_path = "models/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init
    best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                           save_path=model_save_path, **kwargs)
    
    # Plot results
    test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) +\
                ", weight_init:" + init + ", Test Acc: " + str(test_acc)
    best_model.plot_training_progress(show=False,
                                    save=True,
                                    name="figures/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init,
                                    subtitle=subtitle)
    montage(W=np.array(best_model.layers[0].weights[:, :-1]),
            title=subtitle,
            path="figures/" + experiment_name + "/weights/" + dict_to_string(kwargs) + "_" + init)

    # Minimizing value: validation accuracy
    val_acc = best_model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": best_model}  # Save score and model
    return result

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

    search_space = {  # Optimization will be performed on all combinations of these
        "batch_size": [20, 100, 400],     # Batch sizes
        "lr": [0.0001, 0.001, 0.01],      # Learning rates
        "l2_reg": [0.01, 0.05, 0.1],      # L2 Regularization terms
        "momentum" : [0.5, 0.7],
    }
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_name" : "performance_optimization_svm/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "x_test" : x_test,
        "y_test" : y_test,
        "epochs" : 120,
        "init": "fixed"
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path="models/" + fixed_args["experiment_name"])
    best_model = mpo.grid_search(evaluator=evaluator,
                                search_space=search_space,
                                fixed_args=fixed_args)

    # TESTING
    # model = Sequential(loss="cross_entropy")
    # model.add(
    #     Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    # model.add(Activation("softmax"))
    # best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
    #                     batch_size=50, epochs=10, lr=0.01, # 0 lr will not change weights
    #                     momentum=0.5, l2_reg=0.01, save_path="models/performance_optimization/test1")
    # test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    # # subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) + ", Test Acc: " + str(test_acc)
    # best_model.plot_training_progress(show=False,
    #                             save=True,
    #                             name="figures/performance_optimization/test",
    #                             subtitle="subtitle")
    # val_acc = best_model.get_classification_metrics(x_val, y_val)[0]
    # print("test_acc:", test_acc)
    # print("val_acc:", val_acc)
    
    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)


#####################################################
# The file /home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/Assignment_1/performance_optimization.py contains:
 #####################################################
# Add path to Toy-DeepLearning-Framework
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

def montage(W, title, path=None):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    plt.suptitle(title)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    if path is not None:
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()
    

# Define evaluator (function to run in MetaParamOptimizer)
def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", init="fixed", **kwargs):
    # Define model
    model = Sequential(loss="cross_entropy")
    model.add(
        Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization=init))
    model.add(Activation("softmax"))

    # Fit model
    model_save_path = "models/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init
    best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                           save_path=model_save_path, **kwargs)
    
    # Plot results
    test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) +\
                ", weight_init:" + init + ", Test Acc: " + str(test_acc)
    best_model.plot_training_progress(show=False,
                                    save=True,
                                    name="figures/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init,
                                    subtitle=subtitle)
    montage(W=np.array(best_model.layers[0].weights[:, :-1]),
            title=subtitle,
            path="figures/" + experiment_name + "/weights/" + dict_to_string(kwargs) + "_" + init)

    # Minimizing value: validation accuracy
    val_acc = best_model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": best_model}  # Save score and model
    return result

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

    search_space = {  # Optimization will be performed on all combinations of these
        "batch_size": [20, 100, 400],     # Batch sizes
        "lr": [0.0001, 0.001, 0.01],      # Learning rates
        "l2_reg": [0.01, 0.05, 0.1],      # L2 Regularization terms
        "init": ["fixed", "xavier"]       # Weight initialization
    }
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_name" : "performance_optimization/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "x_test" : x_test,
        "y_test" : y_test,
        "epochs" : 100,
        "momentum" : 0.5,
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path="models/" + fixed_args["experiment_name"])
    best_model = mpo.grid_search(evaluator=evaluator,
                                search_space=search_space,
                                fixed_args=fixed_args)

    # TESTING
    # model = Sequential(loss="cross_entropy")
    # model.add(
    #     Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    # model.add(Activation("softmax"))
    # best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
    #                     batch_size=50, epochs=10, lr=0.01, # 0 lr will not change weights
    #                     momentum=0.5, l2_reg=0.01, save_path="models/performance_optimization/test1")
    # test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    # # subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) + ", Test Acc: " + str(test_acc)
    # best_model.plot_training_progress(show=False,
    #                             save=True,
    #                             name="figures/performance_optimization/test",
    #                             subtitle="subtitle")
    # val_acc = best_model.get_classification_metrics(x_val, y_val)[0]
    # print("test_acc:", test_acc)
    # print("val_acc:", val_acc)
    
    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)

