import numpy as np
import matplotlib
import time
import math
from matplotlib import pyplot as plt
import copy

class Dense:
    type = "dense"
    def __init__(self, nodes, input_dim):
        self.nodes = nodes
        self.input_shape = input_dim
        self.weights = np.matrix(np.random.normal(
            # 0, 1./float(self.input_shape),
            0, 1./100.,
            (self.nodes, self.input_shape+1)))  # Add biases
        self.dw = np.zeros(self.weights.shape)  # Weight updates

    def forward(self, inputs):
        # NOTE: Not sure if I need to save input/output
        self.inputs = np.append(inputs, [np.ones(inputs.shape[1])], axis=0) # Add biases
        return self.weights*self.inputs

    def backward(self, in_gradient, lr=0.001, momentum=0.7, l2_regularization=0.1):
        ''' Last layer error = O - T
            Other layers error = next_layer_weights.T*next_
        '''
        # Previous layer error propagation
        left_layer_gradient = -self.weights.T*in_gradient  # TODO: Maybe this should be positive (dont think so)

        # Regularization
        regularization_weights = copy.deepcopy(self.weights)
        regularization_weights[:, -1] = 0  # Set bias column to 0 (no regularization)
        regularization_term = 2*l2_regularization*regularization_weights # Only current layer weights != 0

        # Weight update
        self.dw = momentum*self.dw + (1-momentum)*(in_gradient*self.inputs.T + regularization_term)
        self.weights = -lr*self.dw  # NOTE: Maybe positive
        return left_layer_gradient

 
class Activation:
    type = "activation"
    def __init__(self, activation="softmax"):
        self.activation_name = activation
        self.weights = 0

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = None
        if self.activation_name == "softmax":
            self.outputs = self.__softmax(inputs)
        if self.activation_name == "relu":
            self.outputs = self.__relu(inputs)
        return self.outputs 

    def backward(self, in_gradient, Y_real):
        # NOTE: Why softmx needs Y_real and relu doesnt need anything? (last layer?)
        # Maybe I'm doing something wrong (what if softmax was in the middle?)   
        if self.activation_name == "softmax":
            softmax_diff = np.multiply(np.multiply(self.outputs, Y_real), Y_real - self.outputs)
            return np.multiply(in_gradient, softmax_diff)
        if self.activation_name == "relu":
            # Gradient of activation function evaluated at pre-activation
            return in_gradient * (self.inputs > 0)
        return None

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def __relu(self, x):
        return x * (x > 0)

if __name__ == "__main__":
    layer = Dense(3, 7)
    print(type(layer))
    print(type(layer) == Dense)
    print(type(layer) == Activation)