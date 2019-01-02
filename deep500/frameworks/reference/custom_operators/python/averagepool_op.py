import numpy as np
from deep500.lv0.operators.operator_interface import CustomPythonOp
from deep500.frameworks.reference.custom_operators.python.pool_op_common import get_pad_shape, get_output_shape, pool

class AveragePoolOp(CustomPythonOp):
    def __init__(self, input_descriptors, output_descriptors, auto_pad='NOTSET', count_include_pad=0, kernel_shape=None, pads=None, strides=None):
        super(AveragePoolOp, self).__init__(input_descriptors, output_descriptors)
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors
        
        self.auto_pad = auto_pad
        self.count_include_pad = count_include_pad
        self.kernel_shape = kernel_shape

        temp_pads = []
        temp_strides = []
        for i in range(0, len(kernel_shape)):
            temp_pads.append(0)
            temp_pads.append(0)

            temp_strides.append(1)

        self.pads = temp_pads if pads is None else pads
        self.strides = temp_strides if strides is None else strides

    def forward(self, X):
        inputs = [X]
        if self.auto_pad != 'NOTSET':
            out_shape = get_output_shape(
                self.auto_pad,
                inputs[0].shape[2:],
                self.kernel_shape,
                self.strides
            )
        else:
            input_spatial_shape = inputs[0].shape[2:]
            out_shape = [0] * len(input_spatial_shape)
            for i in range(len(input_spatial_shape)):
                out_shape[i] = int(np.ceil(float(
                    input_spatial_shape[i] - \
                    (self.kernel_shape[i] - 1) + \
                    + self.pads[i] + \
                    self.pads[i + len(self.kernel_shape)]
                    ) / float(self.strides[i]))
                    )

        pad_shape = get_pad_shape(
            self.auto_pad,
            inputs[0].shape[2:],
            self.kernel_shape,
            self.strides,
            out_shape
        )

        pads_computed_before = [] #top, left, ...
        pads_computed_after = [] #bottom, right, ...
        if self.auto_pad == 'SAME_UPPER':
            for i in range(0, len(inputs[0].shape) - 2 ):
                pads_computed_before.append(pad_shape[i] // 2)
                pads_computed_after.append(pad_shape[i] - (pad_shape[i] // 2))
        elif self.auto_pad == 'SAME_LOWER':
            for i in range(0, len(inputs[0].shape) - 2 ):
                pads_computed_before.append(pad_shape[i] - (pad_shape[i] // 2))
                pads_computed_after.append(pad_shape[i] // 2)
        elif self.auto_pad == 'VALID':
            for i in range(0, len(inputs[0].shape) - 2 ):
                pads_computed_before.append(0)
                pads_computed_after.append(0)
        elif self.auto_pad == 'NOTSET':
            for i in range(0, len(inputs[0].shape) - 2 ):
                pads_computed_before.append(self.pads[i * 2])
                pads_computed_after.append(self.pads[i * 2 + 1])
                pad_shape[i] = self.pads[i] + self.pads[i + len(self.kernel_shape)]

        pad_width = [(0, 0), (0, 0)]
        for i in range(0, len(inputs[0].shape) - 2 ):
            pad_width.append((pads_computed_before[i], pads_computed_after[i]))

        if not self.count_include_pad:
            input_padded = np.pad(
                inputs[0],
                pad_width,
                mode='constant',
                constant_values=np.nan
            )
        else:
            input_padded = np.pad(
                inputs[0],
                pad_width,
                mode='constant',
                constant_values=0
            )

        y = pool(
            input_padded,
            inputs[0].shape,
            self.kernel_shape,
            self.strides,
            out_shape,
            pad_shape,
            'AVG',
            count_include_pad=self.count_include_pad)

        return y

    def backward(self, grads, fwd_inputs, fwd_outputs):
        grad_X = np.zeros(fwd_inputs[0].shape, dtype=fwd_inputs[0].dtype)

        out_shape = fwd_outputs[0].shape[2:]

        pad_shape = get_pad_shape(
            self.auto_pad,
            fwd_inputs[0].shape[2:],
            self.kernel_shape,
            self.strides,
            out_shape
        )

        pads_computed_before = [] #top, left, ...
        pads_computed_after = [] #bottom, right, ...
        if self.auto_pad == 'SAME_UPPER':
            for i in range(0, len(fwd_inputs[0].shape) - 2 ):
                pads_computed_before.append(pad_shape[i] // 2)
                pads_computed_after.append(pad_shape[i] - (pad_shape[i] // 2))
        elif self.auto_pad == 'SAME_LOWER':
            for i in range(0, len(fwd_inputs[0].shape) - 2 ):
                pads_computed_before.append(pad_shape[i] - (pad_shape[i] // 2))
                pads_computed_after.append(pad_shape[i] // 2)
        elif self.auto_pad == 'VALID':
            for i in range(0, len(fwd_inputs[0].shape) - 2 ):
                pads_computed_before.append(0)
                pads_computed_after.append(0)
        elif self.auto_pad == 'NOTSET':
            for i in range(0, len(fwd_inputs[0].shape) - 2 ):
                pads_computed_before.append(self.pads[i])
                pads_computed_after.append(self.pads[i + len(self.kernel_shape)])
                pad_shape[i] = self.pads[i] + self.pads[i + len(self.kernel_shape)]

        #avoid out of bounds access, fill with default values
        original_input_shape = np.copy(fwd_inputs[0].shape)
        temp_input_shape = np.copy(fwd_inputs[0].shape)
        temp_output_shape = np.copy(fwd_outputs[0].shape)
        temp_kernel_shape = np.copy(self.kernel_shape)
        temp_pads_computed_before = np.copy(pads_computed_before)
        temp_strides = np.copy(self.strides)
        for i in range(5 - len(fwd_inputs[0].shape)):
            temp_input_shape = np.append(temp_input_shape, [1])
            temp_output_shape = np.append(temp_output_shape, [1])
            temp_kernel_shape = np.append(temp_kernel_shape, [1])
            temp_pads_computed_before = np.append(temp_pads_computed_before, [0])
            temp_strides = np.append(temp_strides, [1])

        grad_X = np.reshape(grad_X, temp_input_shape)
        grad_Y = np.copy(grads[0])
        grad_Y = np.reshape(grad_Y, temp_output_shape)

        #loop over output tensor
        for i in range(temp_output_shape[0]):
            for j in range(temp_output_shape[1]):
                for k in range(temp_output_shape[2]):
                    for l in range(temp_output_shape[3]):
                        for m in range(temp_output_shape[4]):
                            #loop over kernel
                            if self.count_include_pad == 0:
                                temp_n = 0
                                for u in range(temp_kernel_shape[0]):
                                    k_input_temp = - temp_pads_computed_before[0] + k * temp_strides[0] + u
                                    if (k_input_temp >= 0 and k_input_temp < temp_input_shape[2]):
                                        for v in range(temp_kernel_shape[1]):
                                            l_input_temp = - temp_pads_computed_before[1] + l * temp_strides[1] + v
                                            if (l_input_temp >= 0 and l_input_temp < temp_input_shape[3]):
                                                for w in range(temp_kernel_shape[2]):
                                                    m_input_temp = - temp_pads_computed_before[2] + m * temp_strides[2] + w
                                                    if (m_input_temp >= 0 and m_input_temp < temp_input_shape[4]):
                                                        temp_n += 1
                            else :
                                temp_n = 1
                                for u in range(len(self.kernel_shape)) :
                                    temp_n *= self.kernel_shape[u]
                            for u in range(temp_kernel_shape[0]):
                                k_input_temp = - temp_pads_computed_before[0] + k * temp_strides[0] + u
                                if (k_input_temp >= 0 and k_input_temp < temp_input_shape[2]):
                                    for v in range(temp_kernel_shape[1]):
                                        l_input_temp = - temp_pads_computed_before[1] + l * temp_strides[1] + v
                                        if (l_input_temp >= 0 and l_input_temp < temp_input_shape[3]):
                                            for w in range(temp_kernel_shape[2]):
                                                m_input_temp = - temp_pads_computed_before[2] + m * temp_strides[2] + w
                                                if (m_input_temp >= 0 and m_input_temp < temp_input_shape[4]):
                                                        grad_X[i, j, int(k_input_temp), int(l_input_temp), int(m_input_temp)] += \
                                                        1. / temp_n * grad_Y[i, j, k, l, m]

        return [np.reshape(grad_X, original_input_shape)]