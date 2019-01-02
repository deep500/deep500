import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class AddOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(AddOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors    

    def forward(self, A, B):

        return A + B

    def backward(self, grads, fwd_inputs, fwd_outputs):

        grad_fwd_inputs = [grads[0].copy(), grads[0].copy()]

        for i in range(2):
            if fwd_inputs[i].shape != grads[0].shape:
                temp_shape = list(fwd_inputs[i].shape)
                temp_extend = [1] * (len(grads[0].shape)-len(fwd_inputs[i].shape))
                temp_shape = temp_extend + temp_shape

                for j in range(len(temp_shape)):
                    if temp_shape[j] == 1:
                        grad_fwd_inputs[i] = np.sum(grad_fwd_inputs[i], j, keepdims=True)
                grad_fwd_inputs[i] = np.reshape(grad_fwd_inputs[i], fwd_inputs[i].shape)

        return grad_fwd_inputs