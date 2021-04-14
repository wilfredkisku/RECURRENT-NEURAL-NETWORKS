import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def tanh_activation(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def softmax_activation(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def delta_cross_entropy(predicted_output, original_output):
    li = []
    grad = predicted_output
    for i, l in enumerate(original_output):
        if l == 1:
            grad[i] -= 1
    return grad

def tanh_activation_backward(x, top_diff):
    output = np.tanh(x)
    return (1.0 - np.square(output)) * top_diff

def multiplication_backward(weight, x, dz):
    gradient_weight = np.array(np.dot())
    return gradient_weight, chain_gradient
