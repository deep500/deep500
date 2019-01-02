import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class ReshapeOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(ReshapeOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    def forward(self, data, shape):

        return np.reshape(data, shape)

    def backward(self, grads, fwd_inputs, fwd_outputs):
        return [np.reshape(grads[0], fwd_inputs[0].shape), np.zeros(fwd_inputs[1].shape, dtype=np.int64)]
        