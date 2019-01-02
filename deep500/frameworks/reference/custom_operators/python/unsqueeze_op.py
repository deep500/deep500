import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class UnsqueezeOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, axes):
        super(UnsqueezeOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        self.axes = axes

    def forward(self, data):

        expanded = np.copy(data)

        for i in range(len(self.axes)):
            expanded = np.expand_dims(expanded, axis=self.axes[i])

        return expanded
  
    def backward(self, grads, fwd_inputs, fwd_outputs):

        return [np.squeeze(grads[0], axis=tuple(self.axes))]