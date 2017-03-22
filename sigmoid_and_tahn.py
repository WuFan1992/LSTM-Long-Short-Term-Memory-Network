import numpy as np

#define the two activation function

def sigmoid(self,weight_input):

    return 1 / (1+exp(-1*weight_input))


def sigmoid_derivative(self,output):

    x = sigmoid(output)

    return x*(1-x)


def tanh(self,weight_input):

    a = exp(weight_input)

    b = exp(-1*weight_input)

    return (a-b)/(a+b)


def tahn_derivative(self,output):

    c = tahn(output)

    return 1-c^2


