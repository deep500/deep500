import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class GemmOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, alpha=1, beta=1, transA=0, transB=0):
        super(GemmOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, A, B, C):
        A = A.T if self.transA != 0 else A
        B = B.T if self.transB != 0 else B

        return np.matmul(self.alpha * A, B) + self.beta * C

    def backward(self, grads, fwd_inputs, fwd_outputs):
        A = fwd_inputs[0].T if self.transA != 1 else fwd_inputs[0]
        B = fwd_inputs[1].T if self.transB != 1 else fwd_inputs[1]

        grad_A = np.matmul(self.alpha * grads[0], B)
        grad_B = np.matmul(self.alpha * A, grads[0])
        grad_C = np.squeeze(self.beta * grads[0])

        grad_A = grad_A.T if self.transA == 1 else grad_A
        grad_B = grad_B.T if self.transB == 1 else grad_B

        grad_fwd_inputs = [grad_A, grad_B, grad_C]

        for i in range(3):
            if fwd_inputs[i].shape != grads[0].shape:
                temp_shape = list(fwd_inputs[i].shape)
                temp_extend = [1] * (len(grads[0].shape)-len(fwd_inputs[i].shape))
                temp_shape = temp_extend + temp_shape

                for j in range(len(temp_shape)):
                    if temp_shape[j] == 1:
                        grad_fwd_inputs[i] = np.sum(grad_fwd_inputs[i], j, keepdims=True)
                grad_fwd_inputs[i] = np.reshape(grad_fwd_inputs[i], fwd_inputs[i].shape)

        return grad_fwd_inputs