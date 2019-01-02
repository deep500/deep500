import numpy as np

def get_pad_shape(auto_pad,  # type: Text
                  input_spatial_shape,  # type: Sequence[int]
                  kernel_spatial_shape,  # type: Sequence[int]
                  dilations_spatial,
                  strides_spatial,  # type: Sequence[int]
                  output_spatial_shape  # type: Sequence[int]
                  ):  # type: (...) -> Sequence[int]
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                (kernel_spatial_shape[i] + (kernel_spatial_shape[i] - 1) * (dilations_spatial[i] - 1)) - \
                input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape

def get_fullconv_pad_shape(kernel_spatial_shape,
                  dilations_spatial,
                  strides_spatial):
    #both are the same
    fullconv_pad_before = [0] * len(kernel_spatial_shape)
    fullconv_pad_after = [0] * len(kernel_spatial_shape)
    for i in range(len(kernel_spatial_shape)):
        #(dilated filter size - 1)
        temp = (
            (kernel_spatial_shape[i] + \
            (kernel_spatial_shape[i] - 1) * (dilations_spatial[i] - 1) - \
            1)
        )
        fullconv_pad_before[i] = temp
        fullconv_pad_after[i] = temp
        
    return fullconv_pad_before, fullconv_pad_after

def get_output_shape(auto_pad,  # type: Text
                     input_spatial_shape,  # type: Sequence[int]
                     kernel_spatial_shape,  # type: Sequence[int]
                     dilations_spatial,
                     strides_spatial  # type: Sequence[int]
                     ):  # type: (...) -> Sequence[int]
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i]) /
                    float(strides_spatial[i])
                )) - \
                ((kernel_spatial_shape[i] - 1) * (dilations_spatial[i] - 1))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(
                input_spatial_shape[i] - \
                ((kernel_spatial_shape[i] + (kernel_spatial_shape[i] - 1) * dilations_spatial[i]) - 1)) / \
                float(strides_spatial[i])))
    return out_shape

def crosscorrelation(input_spatial_shape,
                     kernel_spatial_shape,
                     group,
                     dilations_spatial,
                     strides_spatial,
                     pads_before,
                     output_spatial_shape,
                     X,
                     W,
                     B):
    #avoiding out of bound accessing
    N = X.shape[0]
    M = W.shape[0]
    original_input_shape = list(X.shape)
    original_weight_shape = list(W.shape)
    original_output_shape = [N, M] +list(output_spatial_shape)
    temp_input_shape = list(X.shape)
    temp_weight_shape = list(W.shape)
    temp_output_shape = [N, M] +  list(output_spatial_shape)
    temp_pads_computed_before = list(pads_before)
    temp_dilations = list(dilations_spatial)
    temp_strides = list(strides_spatial)
    for i in range(5 - len(X.shape)):
        temp_input_shape.append(1)
        temp_weight_shape.append(1)
        temp_output_shape.append(1)
        temp_pads_computed_before.append(0)
        temp_dilations.append(1)
        temp_strides.append(1)

    X = np.reshape(X, temp_input_shape)
    W = np.reshape(W, temp_weight_shape)
    Y = np.zeros(temp_output_shape, dtype=X.dtype)

    for g in range(group):
        #loop over output tensor
        for i in range(temp_output_shape[0]):
            for j in range(int(temp_output_shape[1] / group)):
                j_output_temp = j + g * (temp_output_shape[1] / group)
            #for j in range(temp_output_shape[1]):
                for k in range(temp_output_shape[2]):
                    for l in range(temp_output_shape[3]):
                        for m in range(temp_output_shape[4]):
                            #loop over kenrel
                            for t in range(temp_weight_shape[1]):
                                t_input_temp = t + g * temp_weight_shape[1]
                                for u in range(temp_weight_shape[2]):
                                    k_input_temp = -temp_pads_computed_before[0] + \
                                    k * temp_strides[0] + u * temp_dilations[0]
                                    if (k_input_temp >= 0 and k_input_temp < temp_input_shape[2]):
                                        for v in range(temp_weight_shape[3]):
                                            l_input_temp = -temp_pads_computed_before[1] + \
                                            l * temp_strides[1] + v * temp_dilations[1]
                                            if (l_input_temp >= 0 and l_input_temp < temp_input_shape[3]):
                                                for w in range(temp_weight_shape[4]):
                                                    m_input_temp = -temp_pads_computed_before[2] + \
                                                    m * temp_strides[2] + w * temp_dilations[2]
                                                    if(m_input_temp >= 0 and m_input_temp < temp_input_shape[4]):
                                                        Y[i, int(j_output_temp), k, l, m] += \
                                                        W[int(j_output_temp), t, u, v, w] * \
                                                        X[i, t_input_temp, int(k_input_temp), int(l_input_temp), int(m_input_temp)]
                            Y[i, int(j_output_temp), k, l, m] += B[int(j_output_temp)]
                            
    X = np.reshape(X, original_input_shape)
    W = np.reshape(W, original_weight_shape)

    return np.reshape(Y, original_output_shape)

def crosscorrelation_swap_axes(input_spatial_shape,
                     kernel_spatial_shape,
                     group,
                     dilations_spatial,
                     strides_spatial,
                     pads_before,
                     output_spatial_shape,
                     X,
                     W):

    #avoiding out of bound accessing
    N = W.shape[1]
    M = int(X.shape[1] / group)
    original_input_shape = list(X.shape)
    original_weight_shape = list(W.shape)
    original_output_shape = [N, M] +list(output_spatial_shape)
    temp_input_shape = list(X.shape)
    temp_weight_shape = list(W.shape)
    temp_output_shape = [N, M] +  list(output_spatial_shape)
    temp_pads_computed_before = list(pads_before)
    temp_dilations = list(dilations_spatial)
    temp_strides = list(strides_spatial)
    for i in range(5 - len(X.shape)):
        temp_input_shape.append(1)
        temp_weight_shape.append(1)
        temp_output_shape.append(1)
        temp_pads_computed_before.append(0)
        temp_dilations.append(1)
        temp_strides.append(1)

    X = np.reshape(X, temp_input_shape)
    W = np.reshape(W, temp_weight_shape)
    Y = np.zeros(temp_output_shape, dtype=X.dtype)

    for g in range(group):
        #loop over output tensor
        for i in range(temp_output_shape[1]):
            i_input_temp = i + g * temp_output_shape[1]
            for j in range(int(temp_output_shape[0] / group)):
                j_output_temp = j + g * (temp_output_shape[0] / group)
            
            #for j in range(temp_output_shape[0]):
                for k in range(temp_output_shape[2]):
                    for l in range(temp_output_shape[3]):
                        for m in range(temp_output_shape[4]):
                            #loop over kenrel
                            for t in range(temp_weight_shape[0]):
                                t_input_temp = t + g * temp_weight_shape[0]
                                for u in range(temp_weight_shape[2]):
                                    k_input_temp = -temp_pads_computed_before[0] + \
                                    k * temp_strides[0] + u * temp_dilations[0]
                                    if (k_input_temp >= 0 and k_input_temp < temp_input_shape[2]):
                                        for v in range(temp_weight_shape[3]):
                                            l_input_temp = -temp_pads_computed_before[1] + \
                                            l * temp_strides[1] + v * temp_dilations[1]
                                            if (l_input_temp >= 0 and l_input_temp < temp_input_shape[3]):
                                                for w in range(temp_weight_shape[4]):
                                                    m_input_temp = -temp_pads_computed_before[2] + \
                                                    m * temp_strides[2] + w * temp_dilations[2]
                                                    if(m_input_temp >= 0 and m_input_temp < temp_input_shape[4]):
                                                        Y[int(j_output_temp), i, k, l, m] += \
                                                        W[t, int(j_output_temp), u, v, w] * \
                                                        X[t, i_input_temp, int(k_input_temp), int(l_input_temp), int(m_input_temp)]

    X = np.reshape(X, original_input_shape)
    W = np.reshape(W, original_weight_shape)

    return np.reshape(Y, original_output_shape)

def crosscorrelation_dilx_flipw(
    input_spatial_shape,
    kernel_spatial_shape,
    group,
    dilations_spatial,
    strides_spatial,
    pads_before,
    output_spatial_shape,
    X,
    W,
    X_dilations_spatial):

    N = X.shape[0]
    M = W.shape[1] * group
    original_input_shape = list(X.shape)
    original_weight_shape = list(W.shape)
    original_output_shape = [N, M] + list(output_spatial_shape)
    temp_input_shape = list(X.shape)
    temp_weight_shape = list(W.shape)
    temp_output_shape = [N, M] + list(output_spatial_shape)
    temp_dilations = list(dilations_spatial)
    temp_X_dilations = list(X_dilations_spatial)
    temp_strides = list(strides_spatial)
    temp_pads_before = list(pads_before)
    for i in range(5 - len(X.shape)):
        temp_input_shape.append(1)
        temp_weight_shape.append(1)
        temp_output_shape.append(1)
        temp_pads_before.append(0)
        temp_dilations.append(1)
        temp_X_dilations.append(1)
        temp_strides.append(1)
    
    X = np.reshape(X, temp_input_shape)
    W = np.reshape(W, temp_weight_shape)
    Y = np.zeros(temp_output_shape, dtype=X.dtype)

    #compute dilated imput shape
    temp_dil_inp_shape = [0] * 3
    for i in range(3):
        temp_dil_inp_shape[i] = temp_input_shape[2 + i] + (temp_input_shape[2 + i] - 1) * (temp_X_dilations[i] - 1)


    #weight tensor: swap first and second dimension
    #from M x C x D1 x D2 x D3
    #to C x M x D1 x D2 x D3

    for g in range(group):
            #loop over output tensor
            for i in range(temp_output_shape[0]):
                for j in range(int(temp_output_shape[1] / group)):
                    j_output_temp = j + g * (temp_output_shape[1] / group)
                    for k in range(temp_output_shape[2]):
                        for l in range(temp_output_shape[3]):
                            for m in range(temp_output_shape[4]):
                                #loop over kenrel
                                for t in range(int(temp_weight_shape[0] / group)):
                                    t_input_temp = t + g * (temp_weight_shape[0] / group)
                                    for u in range(temp_weight_shape[2]):
                                        k_input_temp = -temp_pads_before[0] + \
                                        k * temp_strides[0] + u * temp_dilations[0]
                                        if (k_input_temp >= 0 and k_input_temp < temp_dil_inp_shape[0]):
                                            if (k_input_temp % temp_X_dilations[0] == 0):
                                                for v in range(temp_weight_shape[3]):
                                                    l_input_temp = -temp_pads_before[1] + \
                                                    l * temp_strides[1] + v * temp_dilations[1]
                                                    if (l_input_temp >= 0 and l_input_temp < temp_dil_inp_shape[1]):
                                                        if (l_input_temp % temp_X_dilations[1] == 0):
                                                            for w in range(temp_weight_shape[4]):
                                                                m_input_temp = -temp_pads_before[2] + \
                                                                m * temp_strides[2] + w * temp_dilations[2]
                                                                if(m_input_temp >= 0 and m_input_temp < temp_dil_inp_shape[2]):
                                                                    if(m_input_temp % temp_X_dilations[2] == 0):
                                                                        Y[i, int(j_output_temp), k, l, m] += \
                                                                        W[
                                                                            int(t_input_temp),
                                                                            j,
                                                                            temp_weight_shape[2] - u - 1,
                                                                            temp_weight_shape[3] - v - 1,
                                                                            temp_weight_shape[4] - w - 1] * \
                                                                        X[
                                                                            i,
                                                                            int(t_input_temp),
                                                                            int(k_input_temp / temp_X_dilations[0]),
                                                                            int(l_input_temp / temp_X_dilations[1]),
                                                                            int(m_input_temp / temp_X_dilations[2])]

    X = np.reshape(X, original_input_shape)
    W = np.reshape(W, original_weight_shape)

    return np.reshape(Y, original_output_shape)
