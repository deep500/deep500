import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class SqueezeOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, axes):
        super(SqueezeOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

        self.axes = axes

    def forward(self, data):

        return np.squeeze(data, axis=tuple(self.axes))

    def backward(self, grads, fwd_inputs, fwd_outputs):

        grad_data = np.copy(grads[0])

        for i in range(len(self.axes)):
            grad_data = np.expand_dims(grad_data, axis=self.axes[i])

        return [grad_data]