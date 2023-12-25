import math
import numpy as np

class Value:
    """
    A class representing a mathematical value with automatic differentiation capabilities.
    """
    def __init__(self, data, _children=(), _op = "", label=""):
        """
        Initializes a Value object.

        Parameters:
        - data: The numerical value of the object.
        - _children: A tuple of child nodes in the computation graph.
        - _op: The operation associated with the node.
        - label: A label for the node.
        """
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None
        self.grad = 0.0     # Initially the grad is 0, it has no impact

    def __repr__(self) -> str:
        """
        Function that returns a string representation of the Value object.

        Returns:
        A string representing the Value object.
        """
        return f"Value data={self.data}"

    def __add__(self, other):
        """
        Adds two Value objects or a Value object and a scalar.

        Parameters:
        - other: The Value object or scalar to be added.

        Returns:
        A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplies two Value objects or a Value object and a scalar.

        Parameters:
        - other: The Value object or scalar to be multiplied.

        Returns:
        A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        """
        Multiplies a scalar and a Value object.

        Parameters:
        - other: The scalar to be multiplied.

        Returns:
        A new Value object representing the product.
        """
        return self*other

    def __truediv__(self, other): 
        """
        Divides two Value objects or a Value object and a scalar.

        Parameters:
        - other: The Value object or scalar to be divided.

        Returns:
        A new Value object representing the quotient.
        """
        return self * other**-1

    def __neg__(self): 
        """
        Negates the Value object.

        Returns:
        A new Value object representing the negation of the original value.
        """
        return self * -1

    def __sub__(self, other): 
        """
        Subtracts two Value objects or a Value object and a scalar.

        Parameters:
        - other: The Value object or scalar to be subtracted.

        Returns:
        A new Value object representing the difference.
        """
        return self + (-other)

    def __radd__(self, other): 
        """
        Adds a scalar and a Value object.

        Parameters:
        - other: The scalar to be added.

        Returns:
        A new Value object representing the sum.
        """
        return self + other

    def __pow__(self, other):
        """
        Raises the Value object to the power of the specified exponent.

        Parameters:
        - other: The exponent to which the Value object is raised.

        Returns:
        A new Value object representing the result of the exponentiation.
        """
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """
        Computes the exponential function of the Value object.

        Returns:
        A new Value object representing the result of the exponential function.
        """
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Computes the hyperbolic tangent function of the Value object.

        Returns:
        A new Value object representing the result of the hyperbolic tangent function.
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        """
        Applies the ReLU activation function to the Value object.

        Returns:
        A new Value object representing the result of the ReLU activation.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Performs backward propagation to compute gradients.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()