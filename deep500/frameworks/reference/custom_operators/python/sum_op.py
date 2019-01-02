import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class SumOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(SumOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    def forward(self, *inputs):
        temp = np.copy(inputs[0])
        for i in range(1, len(inputs)):
            temp = temp + inputs[i]

        return temp

    def backward(self, grads, fwd_inputs, fwd_outputs):
        num_inputs = len(fwd_inputs)
        grad_fwd_inputs = [grads[0].copy()] * num_inputs

        for i in range(num_inputs):
            if fwd_inputs[i].shape != grads[0].shape:
                temp_shape = list(fwd_inputs[i].shape)
                temp_extend = [1] * (len(grads[0].shape)-len(fwd_inputs[i].shape))
                temp_shape = temp_extend + temp_shape

                for j in range(len(temp_shape)):
                    if temp_shape[j] == 1:
                        grad_fwd_inputs[i] = np.sum(grad_fwd_inputs[i], j, keepdims=True)
                grad_fwd_inputs[i] = np.reshape(grad_fwd_inputs[i], fwd_inputs[i].shape)

        return grad_fwd_inputs