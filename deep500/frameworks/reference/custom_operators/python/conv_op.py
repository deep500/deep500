import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp
from deep500.frameworks.reference.custom_operators.python.conv_op_common import get_pad_shape, get_output_shape, get_fullconv_pad_shape, crosscorrelation, crosscorrelation_dilx_flipw, crosscorrelation_swap_axes
from deep500 import TensorDescriptor

class ConvOp(CustomPythonOp):
    def __init__(
        self,
        input_descriptors,
        output_descriptors,
        auto_pad='NOTSET',
        dilations=None,
        group=1,
        kernel_shape=None,
        pads=None,
        strides=None):
        super(ConvOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

        self.auto_pad = auto_pad
        self.kernel_shape = kernel_shape

        #default values if not specified
        temp_dilations = []
        temp_pads = []
        temp_strides = []
        for i in range(0, len(kernel_shape)):
            temp_dilations.append(1)
            temp_pads.append(0)
            temp_pads.append(0)
            temp_strides.append(1)

        self.dilations = temp_dilations if dilations is None else dilations
        self.group = group
        self.pads = temp_pads if pads is None else pads
        self.strides = temp_strides if strides is None else strides

    def forward(self, X, W, B=None):
        if B is None:
            #optional input B is not given:
            B = np.zeros(W.shape[0], dtype=W.dtype)

        if self.kernel_shape is None:
            self.kernel_shape = W.shape[2:]

        input_spatial_shape = X.shape[2:]
        if self.auto_pad != 'NOTSET':
            out_shape = get_output_shape(
                self.auto_pad,
                X.shape[2:],
                self.kernel_shape,
                self.dilations,
                self.strides
            )
        else:
            
            out_shape = [0] * len(input_spatial_shape)
            for i in range(len(input_spatial_shape)):
                '''
                caffe implementation:
                _const int input_dim = this->input_shape(i + 1);
                const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1; // actual kernel size
                const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                / stride_data[i] + 1;
                '''
                out_shape[i] = int(
                    np.floor(
                        float(
                            input_spatial_shape[i] + \
                            self.pads[i] + \
                            self.pads[i + len(self.kernel_shape)] - \
                            (self.dilations[i] * (self.kernel_shape[i] - 1) + 1)
                        ) / \
                        float(
                            self.strides[i]
                        )
                    ) + 1
                )

        pad_shape = get_pad_shape(
            self.auto_pad,
            X.shape[2:],
            self.kernel_shape,
            self.dilations,
            self.strides,
            out_shape
        )

        pads_computed_before = [] #top, left, ...
        pads_computed_after = [] #bottom, right, ...
        if self.auto_pad == 'SAME_UPPER':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(pad_shape[i] // 2)
                pads_computed_after.append(pad_shape[i] - (pad_shape[i] // 2))
        elif self.auto_pad == 'SAME_LOWER':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(pad_shape[i] - (pad_shape[i] // 2))
                pads_computed_after.append(pad_shape[i] // 2)
        elif self.auto_pad == 'VALID':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(0)
                pads_computed_after.append(0)
        elif self.auto_pad == 'NOTSET':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(self.pads[i])
                pads_computed_after.append(self.pads[i + len(self.kernel_shape)])
                pad_shape[i] = self.pads[i] + self.pads[i + len(self.kernel_shape)]
 
        return crosscorrelation(
            input_spatial_shape,
            self.kernel_shape,
            self.group,
            self.dilations,
            self.strides,
            pads_computed_before,
            out_shape,
            X,
            W,
            B)

    def backward(self, grads, fwd_inputs, fwd_outputs):
        X = fwd_inputs[0]
        W = fwd_inputs[1]
        Y = fwd_outputs[0]
        grad_Y = grads[0]
        if len(fwd_inputs) < 3:
            B = np.zeros(fwd_inputs[1].shape[0], dtype=W.dtype)
        else:
            B = fwd_inputs[2]
        grad_X = np.zeros(X.shape, dtype=X.dtype)
        grad_W = np.zeros(W.shape, dtype=W.dtype)

        #compute pads used in forward:
        pad_shape = get_pad_shape(
            self.auto_pad,
            X.shape[2:],
            self.kernel_shape,
            self.dilations,
            self.strides,
            Y.shape
        )

        pads_computed_before = [] #top, left, ...
        pads_computed_after = [] #bottom, right, ...
        if self.auto_pad == 'SAME_UPPER':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(pad_shape[i] // 2)
                pads_computed_after.append(pad_shape[i] - (pad_shape[i] // 2))
        elif self.auto_pad == 'SAME_LOWER':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(pad_shape[i] - (pad_shape[i] // 2))
                pads_computed_after.append(pad_shape[i] // 2)
        elif self.auto_pad == 'VALID':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(0)
                pads_computed_after.append(0)
        elif self.auto_pad == 'NOTSET':
            for i in range(0, len(X.shape) - 2 ):
                pads_computed_before.append(self.pads[i])
                pads_computed_after.append(self.pads[i + len(self.kernel_shape)])
                pad_shape[i] = self.pads[i] + self.pads[i + len(self.kernel_shape)]


 
        #in order to compute input gradient note:
        #pad for 'full convolution'
        #convolution (crosscorrelation )X * W = Y where W is flipped
        #X = grad_Y
        #W = W
        #dilate W tensor with dilations
        #dilate X tensor with strides
        #no bias

        #compute pads for full convolution
        fullconv_pads_before, fullconv_pads_after = get_fullconv_pad_shape(
            self.kernel_shape,
            self.dilations,
            self.strides)
        for i in range(len(self.kernel_shape)):
            fullconv_pads_before[i] -= pads_computed_before[i]
            fullconv_pads_after[i] -= pads_computed_after[i]
        
        #compute input gradient
        grad_X = crosscorrelation_dilx_flipw(
            grad_Y.shape,
            self.kernel_shape,
            self.group,
            self.dilations,
            [1, 1, 1],
            fullconv_pads_before,
            X.shape[2:],
            grad_Y,
            W,
            self.strides
        )
        
        #in order to compute weight gradient note:
        #swap dilations and strides:
        temp_dilations = list(self.strides)
        temp_strides = list(self.dilations)
        #compute weight gradient, don't use bias
        grad_W = crosscorrelation_swap_axes(
            X.shape[2:],
            Y.shape[2:],
            self.group,
            temp_dilations,
            temp_strides,
            pads_computed_before,
            W.shape[2:],
            X,
            grads[0],
        )

        grad_X = np.reshape(grad_X, X.shape)
        grad_W = np.reshape(grad_W, W.shape)

        if len(fwd_inputs) > 2:
            #compute bias gradient
            grad_B = grad_Y
            for i in range(2, len(Y.shape)):
                grad_B = np.sum(grad_B, axis=2)
            grad_B = np.sum(grad_B, axis=0)
            return [grad_X, grad_W, grad_B]

        else:
            return [grad_X, grad_W]  