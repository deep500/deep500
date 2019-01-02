import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class GlobalAveragePoolOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(GlobalAveragePoolOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    def forward(self, X):
        spatial_shape = np.ndim(X) - 2
        y = np.average(X, axis=tuple(range(spatial_shape, spatial_shape + 2)))
        for _ in range(spatial_shape):
            y = np.expand_dims(y, -1)

        return y

    def backward(self, grads, fwd_inputs, fwd_outputs):
        n = 1
        for i in range(2, len(fwd_inputs[0].shape)):
            n *= fwd_inputs[0].shape[i]

        grad_X = np.full(fwd_inputs[0].shape, 1. / n, dtype=fwd_inputs[0].dtype)

        grad_X = grad_X * grads[0]
        
        return [grad_X]