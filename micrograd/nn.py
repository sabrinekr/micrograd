import random

from value import Value

class Neuron:
    """
    A simple neural network neuron.

    Parameters:
    - nin: Number of input neurons.
    """

    def __init__(self, nin):
        """
        Initializes a Neuron object with random weights and bias.

        Parameters:
        - nin: Number of input neurons.
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Performs a forward pass through the neuron.

        Parameters:
        - x: Input values.

        Returns:
        A Value object representing the output of the neuron after applying the tanh activation.
        """
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        """
        Returns the parameters (weights and bias) of the neuron.

        Returns:
        A list of Value objects representing the weights and bias of the neuron.
        """
        return self.w + [self.b]

class Layer:
    """
    A layer of neurons in a neural network.

    Parameters:
    - nin: Number of input neurons.
    - nout: Number of output neurons.
    """

    def __init__(self, nin, nout):
        """
        Initializes a Layer object with a specified number of input and output neurons.

        Parameters:
        - nin: Number of input neurons.
        - nout: Number of output neurons.
        """
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        Performs a forward pass through the layer.

        Parameters:
        - x: Input values.

        Returns:
        A list of Value objects representing the outputs of the neurons in the layer.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        """
        Returns the parameters of the neurons in the layer.

        Returns:
        A list of Value objects representing the weights and biases of the neurons in the layer.
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (MLP) neural network.

    Parameters:
    - nin: Number of input neurons.
    - nout: List specifying the number of neurons in each layer.
    """

    def __init__(self, nin, nout):
        """
        Initializes an MLP object with a specified number of input neurons and layer sizes.

        Parameters:
        - nin: Number of input neurons.
        - nout: List specifying the number of neurons in each layer.
        """
        sz = [nin] + nout
        self.layers = [Layer(sz[i], sz[i+1])for i in range(len(nout))]

    def __call__(self, x):
        """
        Performs a forward pass through the MLP.

        Parameters:
        - x: Input values.

        Returns:
        A list of Value objects representing the outputs of the MLP.
        """
        for layer in self.layers:
            x = layer(x) 
        return x

    def parameters(self):
        """
        Returns the parameters (weights and biases) of the MLP.

        Returns:
        A list of Value objects representing the weights and biases of the neurons in the MLP.
        """
        return [p for layer in self.layers for p in layer.parameters()]