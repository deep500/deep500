import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class SoftmaxCrossEntropyOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors):
        super(SoftmaxCrossEntropyOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    # TODO: Softmax(X)
    def forward(self, X, target):

        #number of labels
        D = X.shape[1]

        #convert to one-hot fromat, note:
        #one_hot_matrix = np.eye(number_of_labels)[vector]
        target = np.eye(D)[target.astype(np.int64)]
        target = np.reshape(target, X.shape)

        epsilon = 1e-12
        X = np.clip(X, epsilon, 1. - epsilon)
        N = X.shape[0]
        label_cross_entropy = -np.sum(target*np.log(X +1e-9))/N

        return label_cross_entropy

    def backward(self, grads, fwd_inputs, fwd_outputs):

        X = fwd_inputs[0]
        target = fwd_inputs[1]

        #number of labels
        D = X.shape[1]

        #convert to one-hot fromat, note:
        #one_hot_matrix = np.eye(number_of_labels)[vector]
        target = np.eye(D)[target.astype(np.int64)]
        target = np.reshape(target, X.shape)

        grad_target = np.zeros_like(target)

        N = X.shape[0]
        epsilon = 1e-12
        X = np.clip(X, epsilon, 1. - epsilon)
        grad_X = -target * 1 / (X +1e-9)
        grad_X /= N

        return [grad_X, grad_target]