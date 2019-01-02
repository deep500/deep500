import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class CrossEntropyOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(CrossEntropyOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        
    def forward(self, X, target):

        epsilon = 1e-12

        X = np.clip(X, epsilon, 1. - epsilon)
        N = X.shape[0]
        cross_entropy = -np.sum(target*np.log(X +1e-9))/N

        return cross_entropy

    def backward(self, grads, fwd_inputs, fwd_outputs):

        X = fwd_inputs[0]
        target = fwd_inputs[1]

        grad_target = np.zeros_like(target)

        N = X.shape[0]

        epsilon = 1e-12
        X = np.clip(X, epsilon, 1. - epsilon)
        grad_X = -target * 1 / (X +1e-9)
        grad_X /= N

        return [grad_X, grad_target]