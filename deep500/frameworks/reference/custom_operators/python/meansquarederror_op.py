import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class MeanSquaredErrorOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(MeanSquaredErrorOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        
    def forward(self, X, target):

        # calculate the actual loss
        mse = np.mean(((X - target) ** 2), axis=1)
        # then average over batches
        mse_mean = np.mean(mse)

        return mse_mean

    def backward(self, grads, fwd_inputs, fwd_outputs):

        X = fwd_inputs[0]
        target = fwd_inputs[1]

        grad_X = 2 * (X - target) / X.size
        grad_target = np.zeros_like(target)

        return [grad_X, grad_target]