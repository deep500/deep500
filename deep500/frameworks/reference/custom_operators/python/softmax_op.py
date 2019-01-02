import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class SoftmaxOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, axis=1):
        super(SoftmaxOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

        self.axis = axis

    def forward(self, input):
        #numericlly stable version of softmax function

        #compute N = batch size
        N = 1
        for i in range(0, self.axis):
            N = N * input.shape[i]

        #compute D = input feature dimensions
        D = 1
        for i in range(self.axis, len(input.shape)):
            D = D * input.shape[i]

        #coerce input tensor into a 2D matrix of size N*D
        x = np.reshape(input, (N, D))

        max_x = np.max(x, axis=1).reshape((-1, 1))

        exp_x = np.exp(x - max_x)

        return np.reshape((exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))), input.shape)

    def backward(self, grads, fwd_inputs, fwd_outputs):
        #https://software.intel.com/en-us/daal-programming-guide-softmax-backward-layer

        N = 1
        for i in range(0, self.axis):
            N = N * fwd_inputs[0].shape[i]

        #compute D = input feature dimensions
        D = 1
        for i in range(self.axis, len(fwd_inputs[0].shape)):
            D = D * fwd_inputs[0].shape[i]

        #coerce output tensor into a 2D matrix of size N*D
        y = np.reshape(fwd_outputs[0], (N, D))

        #coerce bprop_nextlayer tensor into a 2D matrix of size N*D
        grad_y = np.reshape(grads[0], (N, D))

        sum_grad_y = np.sum(grad_y * y, axis=1).reshape((-1, 1))

        grad_x = np.reshape((y * (grad_y - sum_grad_y)), fwd_inputs[0].shape)

        return [grad_x]