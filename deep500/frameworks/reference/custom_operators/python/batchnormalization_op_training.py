import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp

class BatchNormalizationTrainingOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, epsilon=1e-05, momentum=0.9, spatial=1, is_test=False):
        super(BatchNormalizationTrainingOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        
        self.epsilon = epsilon
        self.momentum = momentum
        self.spatial = spatial

    def forward(self, X, scale, B, mean, var):

        #N = number of elements in minibatch per channel
        N = X.shape[0]
        for i in range(2, len(X.shape)):
            N = N * X.shape[i]

        #compute mean
        saved_mean = np.copy(X)
        for i in range(2, len(X.shape)):
            saved_mean = np.sum(saved_mean, axis=i, keepdims=True)
        saved_mean = np.sum(saved_mean, axis=0)
        saved_mean = (1. / N) * saved_mean

        #compute variance
        xmu = X - saved_mean
        sq = xmu ** 2
        saved_var= np.copy(sq)
        for i in range(2, len(sq.shape)):
            saved_var = np.sum(saved_var, axis=i, keepdims=True)
        saved_var = np.sum(saved_var, axis=0)
        saved_var = (1. / N) * saved_var
         
        #compute running mean
        saved_mean = np.squeeze(saved_mean)
        running_mean = mean * self.momentum + saved_mean * (1 - self.momentum)

        #compute running variance
        saved_var = np.squeeze(saved_var)
        running_var = var * self.momentum + saved_var * (1 - self.momentum) 

        dims_x = len(X.shape)
        dim_ones = (1,) * (dims_x - 2)
        scale = scale.reshape(-1, *dim_ones)
        B = B.reshape(-1, *dim_ones)
        saved_mean = saved_mean.reshape(-1, *dim_ones)
        saved_var = saved_var.reshape(-1, *dim_ones)

        out = scale * (X - saved_mean) / np.sqrt(saved_var + self.epsilon) + B

        return [
            out,
            running_mean,
            running_var,
            np.reshape(saved_mean, running_mean.shape),
            np.reshape(saved_var, running_var.shape)]

    def backward(self, grads, fwd_inputs, fwd_outputs):
        #similar to source: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        x = fwd_inputs[0]
        saved_mean = fwd_outputs[3]
        saved_var = fwd_outputs[4]
        
        dims_x = len(x.shape)
        dim_ones = (1,) * (dims_x - 2)

        #N = number of elements in minibatch per channel
        N = x.shape[0]
        for i in range(2, len(x.shape)):
            N = N * x.shape[i]

        saved_mean = saved_mean.reshape(-1, *dim_ones)

        xmu = x - saved_mean

        saved_var = saved_var.reshape(-1, *dim_ones)

        sqrtvar = np.sqrt(saved_var + self.epsilon)
        
        ivar = 1./sqrtvar

        xhat = xmu * ivar

        dbeta = np.copy(grads[0])
        for i in range(2, len(grads[0].shape)):
            dbeta = np.sum(dbeta, axis=i, keepdims=True)
        dbeta = np.sum(dbeta, axis=0, keepdims=True)
        
        dgammax = grads[0]
        dgamma = np.copy(dgammax*xhat)
        for i in range(2, len((dgammax*xhat).shape)):
            dgamma = np.sum(dgamma, axis=i, keepdims=True)
        dgamma = np.sum(dgamma, axis=0, keepdims=True)
        
        scale = np.copy(fwd_inputs[1])
        scale = scale.reshape(-1, *dim_ones)
        
        dxhat = dgammax * scale

        divar = np.copy(dxhat*xmu)
        for i in range(2, len((dxhat*xmu).shape)):
            divar = np.sum(divar, axis=i, keepdims=True)
        divar = np.sum(divar, axis=0)
        
        dxmu1 = dxhat * ivar

        dsqrtvar = -1. /(sqrtvar**2) * divar

        dvar = 0.5 * 1. /np.sqrt(saved_var+self.epsilon) * dsqrtvar

        dsq = 1. /N * np.ones(fwd_inputs[0].shape) * dvar

        dxmu2 = 2 * xmu * dsq

        dx1 = (dxmu1 + dxmu2)
        
        dmu = np.copy(dxmu1 + dxmu2)
        for i in range(2, len((dxmu1 + dxmu2).shape)):
            dmu = np.sum(dmu, axis=i, keepdims=True)
        dmu = np.sum(dmu, axis=0)
        dmu = -1 * dmu

        dx2 = 1. /N * np.ones(fwd_inputs[0].shape) * dmu

        dx = dx1 + dx2

        return [
            dx.reshape(fwd_inputs[0].shape), 
            dgamma.reshape(fwd_inputs[1].shape),
            dbeta.reshape(fwd_inputs[2].shape),
            np.zeros_like(fwd_inputs[3]),
            np.zeros_like(fwd_inputs[4])]