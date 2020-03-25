import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))

from layers import Activation, Dense
from utils import prob_to_class, accuracy

class Sequential:
    def __init__(self, loss="cross_entropy", reg_term=0.1):
        self.layers = []
        self.loss = loss
        self.reg_term = reg_term

    def add(self, layer):
        """Add layer"""
        self.layers.append(layer)

    def predict(self, X):
        """Forward pass"""
        vals = X
        for layer in self.layers:
            vals = layer.forward(vals)
        return vals

    def get_metrics(self, X, Y_real):
        if X is None or Y_real is None:
            return 1, 0
        Y_pred_prob = self.predict(X)
        Y_pred_classes = prob_to_class(Y_pred_prob)
        acc = accuracy(Y_pred_classes, Y_real)
        loss = self.__loss(Y_pred_prob, Y_real)
        return acc, loss

    def __cross_entropy(self, Y_pred, Y_real):
        return -np.sum(np.log(np.sum(np.multiply(Y_pred, Y_real), axis=1)))

    def __loss(self, Y_pred_prob, Y_real):
        if self.loss == "cross_entropy":
            return self.__cross_entropy(Y_pred_prob, Y_real)
        return None

    def __loss_differential(self, Y_pred, Y_real):
        if self.loss == "cross_entropy":
            # d (-log(x))/dx = -1/x 
            f_y = np.multiply(Y_real, Y_pred)
            loss_diff = -np.reciprocal(f_y, out=np.zeros_like(Y_pred), where=f_y!=0)  # Element-wise inverse
            return loss_diff
            # G = - (Y_real - Y_pred)
            # return G*X.T/X.shape[1]; 
        return None

    def __cost(self, Y_pred_prob, Y_real):
        """Computes cost = loss + regularization"""
        # Loss
        loss = 0
        if self.loss == "cross_entropy":
            loss = self.__cross_entropy(Y_pred_prob, Y_real)

        # Regularization
        w_norm = 0
        for layer in self.layers:
            w_norm += np.linalg.norm(layer.weights, 'fro')**2

        return loss/Y_pred_prob.shape[1] + self.reg_term*w_norm

    def fit(self, X, Y, X_val=None, Y_val=None, batch_size=None, epochs=100, lr=0.01, momentum=0.7, l2_reg=0.1):
        if batch_size is None or batch_size > X_val.shape[1]:
            batch_size = X_val.shape[1]
        
        # Learning tracking
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []


        print(self.layers)

        for epoch in tqdm(range(epochs)):
            indx = list(range(X.shape[1])) 
            np.random.shuffle(indx)
            for i in range(int(X.shape[1]/batch_size)):  # Missing last X.shape[1]%batch_size but should be ok
                # Get minibatch
                X_minibatch = X[:, i:i+batch_size]
                Y_minibatch = Y[:, i:i+batch_size]
                
                # Forward pass
                Y_pred_prob = self.predict(X_minibatch)

                # Backprop
                gradient = self.__loss_differential(Y_pred_prob, Y_minibatch)  # First error id (D loss)/(D weight)
                for layer in reversed(self.layers):  # Next errors given by each layer weights
                    print("--", type(layer))
                    print(isinstance(layer, Dense))
                    if type(layer) == Activation:  # Activation layers dont need to update weights
                        print(type(layer))
                        gradient = layer.backward(
                                        in_gradient=gradient,
                                        Y_real=Y_minibatch)
                    elif type(layer) == Dense:
                        print(type(layer))
                        gradient = layer.backward(
                                        in_gradient=gradient,
                                        lr=lr,
                                        momentum=momentum,
                                        l2_regularization=l2_reg)[:-1, :]  # Remove bias

            # Error tracking:
            train_acc, train_loss = self.get_metrics(X, Y)
            val_acc, val_loss = self.get_metrics(X_val, Y_val)
            self.train_loss.append(train_acc)
            self.val_loss.append(train_loss)
            self.train_accuracy.append(val_acc)
            self.val_accuracy.append(val_loss)


    def plot_training_progress(self, show=True, save=False, name="model_results"):
        plt.figure()
        plt.plot(list(range(len(self.train_errors))),
                 self.train_errors, label="Train loss", c="orange")
        if len(self.val_errors) > 0:
            plt.plot(list(range(len(self.val_errors))),
                     self.val_errors, label="Val loss", c="red")
        n = len(self.train_acc)
        plt.plot(list(range(n)),
                 np.array(self.train_acc), label="Train acc", c="green")
        if len(self.val_acc) > 0:
            n = len(self.val_acc)
            plt.plot(list(range(n)),
                     np.array(self.val_acc), label="Val acc", c="blue")
        plt.title("Training Evolution")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Metrics")
        if save:
            plt.savefig("figures/" + name + ".png")
        if show:
            plt.show()
