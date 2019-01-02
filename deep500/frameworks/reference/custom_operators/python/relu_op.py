import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class ReluOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(ReluOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    def forward(self, X):

        return np.clip(X, 0, np.inf)

    def backward(self, grads, fwd_inputs, fwd_outputs):
        mask = fwd_inputs[0] < 0
        d_input = np.copy(grads[0])
        d_input[mask] = 0
        
        d_input = np.reshape(d_input, fwd_inputs[0].shape)

        return [d_input]