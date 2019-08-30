import itertools
from functools import partial

import numpy as np
import tensorflow as tf

import deep500 as d5

from .tf_network import TensorflowNetwork


class TensorflowVisitor(d5.OnnxBaseVisitor):

    def __init__(self):
        self.counter = 0
        self.net_input = {}
        self.is_training = None

    def visit_graph(self, graph: d5.ops.OnnxGraph, network: TensorflowNetwork):
        self.net_input.clear()
        tf.reset_default_graph()
        self.is_training = tf.placeholder(tf.bool)

    def visit_net_input(self, input: d5.ops.OnnxValueInfo, network: TensorflowNetwork):
        if isinstance(input.type, d5.ops.OnnxTensorType):
            tensor_type = input.type
            self.net_input[input.name] = (tensor_type.type.to_numpy(), tensor_type.shape.shape)
        else:
            raise NotImplementedError('Only tensor input supported currently')

    def visit_net_output(self, output: d5.ops.OnnxValueInfo, network: TensorflowNetwork):
        network.output_names.append(output.name)

    def visit_initializer(self, initializer: d5.ops.OnnxTensor, network: TensorflowNetwork):
        network.feed_tensor(initializer.name, initializer.data, is_param=True)

        if initializer.name in self.net_input:
            del self.net_input[initializer.name]

    def visit_constant(self, op: d5.ops.Constant, network: TensorflowNetwork):
        tensor = tf.convert_to_tensor(op.value.get_value())
        network.feed_internal_tensor(op.o_output, tensor)

    def visit_initializer_end(self, network: TensorflowNetwork):
        for name, (numpy_type, shape) in self.net_input.items():
            placeholder = tf.placeholder(dtype=numpy_type, shape=shape, name=name)
            network.feed_internal_tensor(name, placeholder)

    def visit_add(self, op: d5.ops.Add, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.add(A, B)

        network.feed_internal_tensor(op.o_C, C)

    def visit_dropout(self, op: d5.ops.Dropout, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        ratio = op.ratio.get_value() if op.ratio else 0.5
        Y = tf.layers.dropout(X, rate=ratio, training=self.is_training)
        network.feed_internal_tensor(op.o_output, Y)

    def visit_sub(self, op: d5.ops.Sub, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.subtract(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_mul(self, op: d5.ops.Mul, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.multiply(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_xor(self, op: d5.ops.Xor, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.logical_xor(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_div(self, op: d5.ops.Div, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.div(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_equal(self, op: d5.ops.Equal, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.equal(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_greater(self, op: d5.ops.Greater, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.greater(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_less(self, op: d5.ops.Less, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.less(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_or(self, op: d5.ops.Or, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        C = tf.logical_or(A, B)
        network.feed_internal_tensor(op.o_C, C)

    def visit_not(self, op: d5.ops.Not, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.logical_not(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_argmax(self, op: d5.ops.ArgMax, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        Y = tf.argmax(X)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_argmin(self, op: d5.ops.ArgMin, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        Y = tf.argmin(X)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_floor(self, op: d5.ops.Floor, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.floor(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_cast(self, op: d5.ops.Cast, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_input)
        Y = tf.cast(X, op.to.get_value())
        network.feed_internal_tensor(op.o_output, Y)

    def visit_affine(self, op: d5.ops.Affine, network: TensorflowNetwork):
        # y = alpha * x + beta,
        X = network.fetch_internal_tensor(op.i_X)
        Y = op.alpha.get_value() * X + op.beta.get_value()
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_ceil(self, op: d5.ops.Ceil, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.ceil(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_reducel1(self, op: d5.ops.ReduceL1, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        axis = [int(v) for v in axis] if isinstance(axis, list) else axis
        if len(axis) == 1:
            axis = axis[0]
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.norm(X, ord=1, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducel2(self, op: d5.ops.ReduceL2, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        axis = [int(v) for v in axis] if isinstance(axis, list) else axis
        if len(axis) == 1:
            axis = axis[0]
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.norm(X, ord=2, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducesumsquare(self, op: d5.ops.ReduceSumSquare, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_sum(tf.square(X), axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducelogsum(self, op: d5.ops.ReduceLogSum, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.log(tf.reduce_sum(X, axis=axis, keepdims=keepdims))
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducesum(self, op: d5.ops.ReduceSum, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_sum(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducemean(self, op: d5.ops.ReduceMean, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_mean(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducemax(self, op: d5.ops.ReduceMax, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_max(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducemin(self, op: d5.ops.ReduceMin, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_min(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reduceprod(self, op: d5.ops.ReduceProd, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_prod(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_reducelogsumexp(self, op: d5.ops.ReduceLogSumExp, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_data)
        axis = op.axes.get_value() if op.axes is not None else list(range(len(X.get_shape().as_list())))
        keepdims = True if op.keepdims is None else op.keepdims.get_value() == 1
        Y = tf.reduce_logsumexp(X, axis=axis, keepdims=keepdims)
        network.feed_internal_tensor(op.o_reduced, Y)

    def visit_relu(self, op: d5.ops.Relu, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.nn.relu(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_max(self, op: d5.ops.Max, network: TensorflowNetwork):
        X = network.fetch_internal_tensors(op.input)
        Y = tf.reduce_max(tf.stack(X), axis=0)
        network.feed_internal_tensor(op.o_max, Y)

    def visit_min(self, op: d5.ops.Min, network: TensorflowNetwork):
        X = network.fetch_internal_tensors(op.input)
        Y = tf.reduce_min(tf.stack(X), axis=0)
        network.feed_internal_tensor(op.o_min, Y)

    def visit_mean(self, op: d5.ops.Mean, network: TensorflowNetwork):
        X = network.fetch_internal_tensors(op.input)
        Y = tf.reduce_mean(tf.stack(X), axis=0)
        network.feed_internal_tensor(op.o_mean, Y)

    def visit_prelu(self, op: d5.ops.PRelu, network: TensorflowNetwork):
        X, slope = network.fetch_internal_tensors([op.i_X, op.i_slope])
        slope = self.expand_to_broadcast(slope, 1, len(X.get_shape()))
        pos = tf.nn.relu(X)
        neg = slope * (X - abs(X)) * 0.5
        Y = pos + neg
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_leakyrelu(self, op: d5.ops.LeakyRelu, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        alpha = op.alpha.get_value() if op.alpha else 0.01
        Y = tf.nn.relu(X) - alpha * tf.nn.relu(-X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_slice(self, op: d5.ops.Slice, network: TensorflowNetwork):
        # Adapted from:
        # https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/backends/backend_v1.py#L700
        X = network.fetch_internal_tensor(op.i_data)
        X_shape = X.get_shape().as_list()
        X_begin = [0] * len(X_shape)
        starts = op.starts.get_value()
        ends = op.ends.get_value()
        slice_len = len(starts)
        axes = op.axes.get_value() if op.axes else list(range(slice_len))

        for i in range(slice_len):
            ends[i] = X_shape[axes[i]] + ends[i] if ends[i] < 0 else ends[i]
            if X_shape[axes[i]] is not None:
                ends[i] = np.min([X_shape[axes[i]], ends[i]])
                starts[i] = np.min([X_shape[axes[i]], starts[i]])
            X_begin[axes[i]] = starts[i]
            X_shape[axes[i]] = ends[i] - starts[i]

        Y = tf.slice(X, tf.constant(X_begin), tf.constant(X_shape))
        network.feed_internal_tensor(op.o_output, Y)

    def visit_clip(self, op: d5.ops.Clip, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_input)

        max = op.max.get_value() if op.max else tf.reduce_max(X)
        min = op.min.get_value() if op.min else tf.reduce_min(X)

        Y = tf.clip_by_value(X, min, max)
        network.feed_internal_tensor(op.o_output, Y)

    def visit_sum(self, op: d5.ops.Sum, network: TensorflowNetwork):
        X = network.fetch_internal_tensors(op.input)
        Y = tf.reduce_sum(X, axis=0)
        network.feed_internal_tensor(op.o_sum, Y)

    def visit_abs(self, op: d5.ops.Abs, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.abs(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_neg(self, op: d5.ops.Neg, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        Y = tf.negative(X)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_pow(self, op: d5.ops.Pow, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_X, op.i_Y])
        C = tf.pow(A, B)
        network.feed_internal_tensor(op.o_Z, C)

    def visit_reshape(self, op: d5.ops.Reshape, network: TensorflowNetwork):
        X, Shape = network.fetch_internal_tensors([op.i_data, op.i_shape])
        Y = tf.reshape(X, Shape)
        network.feed_internal_tensor(op.o_reshaped, Y)

    def expand_to_broadcast(self, X, broadcast_dim=1, total_num_dim=4):
        if broadcast_dim < 0:
            broadcast_dim += total_num_dim
        dims = [broadcast_dim + i for i in range(len(X.shape))]
        for i in range(total_num_dim):
            if i not in dims:
                X = tf.expand_dims(X, i)
        return X

    def visit_and(self, op: d5.ops.And, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        Y = tf.logical_and(A, B)
        network.feed_internal_tensor(op.o_C, Y)

    def visit_softmax(self, op: d5.ops.Softmax, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_input)
        Y = tf.nn.softmax(X)
        network.feed_internal_tensor(op.o_output, Y)

    def visit_cross_entropy(self, op: d5.ops.CrossEntropy, network: TensorflowNetwork):
        labels = tf.placeholder(tf.int32, name=op.i_target)
        network.feed_internal_tensor(op.i_target, labels)

        X = network.fetch_internal_tensor(op.i_X)
        L = -tf.reduce_sum(labels * tf.log(X), 1)
        L = tf.reduce_mean(L, axis=0)
        network.loss_gradient = L
        network.feed_internal_tensor(op.o_output, L)
        network.add_output(op.o_output)

    def visit_softmax_cross_entropy(self, op: d5.ops.SoftmaxCrossEntropy, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)

        labels = tf.placeholder(tf.int32, name=op.i_target)
        network.feed_internal_tensor(op.i_target, labels)

        labels = tf.one_hot(labels, X.get_shape().as_list()[-1])
        result = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels,
            X,
            axis=None  # Defaults to -1
        )
        L = tf.reduce_mean(result, axis=0)
        network.loss_gradient = L
        network.feed_internal_tensor(op.o_output, L)
        network.add_output(op.o_output)

    def visit_mean_squared_error(self, op: d5.ops.MeanSquaredError, network: TensorflowNetwork):
        y_pred = network.fetch_internal_tensor(op.i_X)

        y_true = tf.placeholder(tf.float32, name=op.i_target, shape=y_pred.shape)
        network.feed_internal_tensor(op.i_target, y_true)

        L = tf.nn.l2_loss(y_true - y_pred)

        network.loss_gradient = L
        network.feed_internal_tensor(op.o_output, L)
        network.add_output(op.o_output)

    def visit_matmul(self, op: d5.ops.MatMul, network: TensorflowNetwork):
        A, B = network.fetch_internal_tensors([op.i_A, op.i_B])
        Y = tf.matmul(A, B)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_pad(self, op, network):
        data = network.fetch_internal_tensor(op.i_data)
        if all([p == 0 for p in op.pads.value]): # No padding?
            network.feed_internal_tensor(op.o_output, data)
            return

        y = tf.pad(
            data,
            op.pads.value,
            mode=op.mode.value,
            constant_values=op.value.value
        )
        network.feed_internal_tensor(op.o_output, y)

    def visit_shape(self, op, network):
        data = network.fetch_internal_tensor(op.i_data)
        network.feed_internal_tensor(op.o_shape, tf.shape(data, out_type=tf.int64))

    def visit_squeeze(self, op, network):
        data = network.fetch_internal_tensor(op.i_data)
        network.feed_internal_tensor(op.o_squeezed, 
                                     tf.squeeze(data, axis=op.axes.value))

    def visit_unsqueeze(self, op, network):
        data = network.fetch_internal_tensor(op.i_data)
        result = data
        for axis in op.axes.value:
            result = tf.expand_dims(result, axis=axis)
        network.feed_internal_tensor(op.o_expanded, result)

    def visit_concat(self, op, network):
        in_tensors = [network.fetch_internal_tensor(i) for i in op.input]
        out = tf.concat(in_tensors, op.axis.value)
        network.feed_internal_tensor(op.output[0], out)


    def visit_lrn(self, op, network):
        X = network.fetch_internal_tensor(op.i_X)
        nsize = (op.size.value - 1) // 2
        result = tf.nn.local_response_normalization(
            X,
            depth_radius=nsize,
            bias=op.bias.value,
            alpha=op.alpha.value / op.size.value,
            beta=op.beta.value)
        network.feed_internal_tensor(op.o_Y, result)

    def visit_split(self, op, network):
        input = network.fetch_internal_tensor(op.i_input)
        results = tf.split(
            input,
            op.split.value,
            axis=op.axis.value)
        for res,out in zip(results, op.output):
            network.feed_internal_tensor(out, res)

    def visit_gather(self, op, network):
        input = network.fetch_internal_tensor(op.i_data)
        indices = network.fetch_internal_tensor(op.i_indices)
        results = tf.gather(
            input,
            indices,
            axis=op.axis.value)
        network.feed_internal_tensor(op.o_output, results)

    def visit_gemm(self, op: d5.ops.Gemm, network: TensorflowNetwork):
        (A, B, C) = network.fetch_internal_tensors([op.i_A, op.i_B, op.i_C])
        alpha = 1.0 if op.alpha is None else op.alpha.get_value()
        beta = 1.0 if op.beta is None else op.beta.get_value()
        trans_a = 0 if op.transA is None else op.transA.get_value()
        trans_b = 0 if op.transB is None else op.transB.get_value()

        if trans_a:
            A = tf.transpose(A)
        if trans_b:
            B = tf.transpose(B)
        Y = alpha * tf.matmul(A, B) + beta * C

        network.feed_internal_tensor(op.o_Y, Y)

    def visit_flatten(self, op: d5.ops.Flatten, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_input)
        shape = tf.shape(X)
        x_rank = len(X.shape)
        axis = 1 if op.axis is None else op.axis.get_value()

        if axis == 1 and x_rank > 1:
            Y = tf.layers.flatten(X)
        else:
            if axis == 0:
                cal_shape = (1, -1)
            else:
                cal_shape = (tf.reduce_prod(shape[0:axis]),
                             tf.reduce_prod(shape[axis:tf.size(shape)]))
            Y = tf.reshape(X, cal_shape)

        network.feed_internal_tensor(op.o_output, Y)

    def visit_globalmaxpool(self, op: d5.ops.GlobalMaxPool, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        dims = tf.range(tf.rank(X))
        _, dim_window = tf.split(dims, [2, tf.size(2) - 2])
        Y = tf.reduce_max(X, axis=dim_window, keep_dims=True)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_maxpool(self, op: d5.ops.MaxPool, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        kernel_shape = op.kernel_shape.get_value()
        strides = None if op.strides is None else op.strides.get_value()
        auto_pad = None if op.auto_pad is None else op.auto_pad.get_value()
        pads = None if op.pads is None else op.pads.get_value()
        Y = self.pool(X, kernel_shape, partial(tf.nn.pool, pooling_type='MAX'), 'MAX',
                      strides=strides,
                      pads=pads,
                      count_include_pad=None,
                      auto_pad=auto_pad)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_averagepool(self, op: d5.ops.AveragePool, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)
        kernel_shape = op.kernel_shape.get_value()
        strides = None if op.strides is None else op.strides.get_value()
        auto_pad = None if op.auto_pad is None else op.auto_pad.get_value()
        pads = None if op.pads is None else op.pads.get_value()
        Y = self.pool(X, kernel_shape, partial(tf.nn.pool, pooling_type='AVG'), 'AVG',
                      strides=strides,
                      pads=pads,
                      count_include_pad=op.count_include_pad,
                      auto_pad=auto_pad)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_globalaveragepool(self, op: d5.ops.GlobalAveragePool, network: TensorflowNetwork):
        X = network.fetch_internal_tensor(op.i_X)

        modtyp = getattr(tf.keras.layers,
                         'GlobalAveragePooling%dD' % (len(X.shape) - 2), None)
        if modtyp is None:
            raise RuntimeError('Unsupported dimensions for global average pool'
                               '(%d)' % (len(X.shape) - 2))

        # ONNX forces channels_first format
        tfop = modtyp(data_format='channels_first')

        # Spatial mean w.r.t. channel dimensions
        Y = tfop.apply(X)

        network.feed_internal_tensor(op.o_Y, Y)

    def visit_conv(self, op: d5.ops.Conv, network: TensorflowNetwork):
        X, W = network.fetch_internal_tensors([op.i_X, op.i_W])
        bias = op.i_B is not None
        B = None
        if bias:
            B = network.fetch_internal_tensor(op.i_B)

        kernel_shape = op.kernel_shape.get_value() if op.kernel_shape else None
        pads = op.pads.get_value() if op.pads else None
        strides = op.strides.get_value() if op.strides else None
        dilations = op.dilations.get_value() if op.dilations else None
        group = op.group.get_value() if op.group else None
        Y = self._conv(X, W, kernel_shape is not None, B=B, kernel_shape=kernel_shape, dilations=dilations,
                       strides=strides, pads=pads, group=group)
        network.feed_internal_tensor(op.o_Y, Y)

    def visit_convtranspose(self, op: d5.ops.ConvTranspose, network: TensorflowNetwork):
        X, W = network.fetch_internal_tensors([op.i_X, op.i_W])
        bias = op.i_B is not None
        B = None
        if bias:
            B = network.fetch_internal_tensor(op.i_B)

        kernel_shape = op.kernel_shape.get_value() if op.kernel_shape else None
        pads = op.pads.get_value() if op.pads else None
        strides = op.strides.get_value() if op.strides else None
        dilations = op.dilations.get_value() if op.dilations else None
        group = op.group.get_value() if op.group else None
        output_padding = op.output_padding.get_value() if op.output_padding else None
        output_shape = op.output_shape.get_value() if op.output_shape else None
        Y = self._conv(X, W, kernel_shape is not None, B=B, kernel_shape=kernel_shape, dilations=dilations,
                       strides=strides, pads=pads, group=group, output_padding=output_padding,
                       output_shape=output_shape, transpose=True)
        network.feed_internal_tensor(op.o_Y, Y)

    # Taken from https://github.com/onnx/onnx-tensorflow/
    def _conv_(self, x, in_weights, kernel_shape, pads, strides, dilations, group, transpose, is_bias=False, b=None,
               output_shape=None, output_padding=None):
        """ Convolution method for both conv and transposed conv
        For transposed conv,
          Attr pads is not used for input, but declares how much output is padded.
          Here, output means output from transposed conv which already pad output_padding if set.
          So the pseudo explanation for output should be:
            output = conv_transpoe_output + output_padding - pads
          And conv_transpoe_output shape should be:
            conv_transpoe_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
        """
        x_rank = len(x.get_shape())
        x_shape = x.get_shape().as_list()
        spatial_size = x_rank - 2

        support_cuda = self.supports_device("CUDA")
        storage_format, compute_format = self.get_data_format(x_rank, support_cuda)
        compute_c_idx = compute_format.find("C")
        spatial_format = "".join([d for d in compute_format if d not in ["N", "C"]])

        weights_rank = len(in_weights.get_shape())
        if transpose:
            # Translate weights from (C x M x KH x KW) to (KH x KW X M X C)
            perm = list(range(2, weights_rank)) + [1, 0]
        else:
            # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
            perm = list(range(2, weights_rank)) + [1, 0]

        assert in_weights.get_shape().as_list()[2:] == kernel_shape, (
            "kernel_shape "
            "attr of convolution does not match the actual weight "
            "passed to this operation, attr {}, actual {}").format(
            kernel_shape,
            in_weights.get_shape().as_list())

        weights = tf.transpose(in_weights, perm)

        pads = pads if pads else [0, 0] * spatial_size

        if not transpose:
            x = self.get_padding_as_op(x, pads)

        group = group if group else 1

        weight_groups = tf.split(weights, num_or_size_splits=group, axis=-1)

        if support_cuda:
            xs = tf.split(x, num_or_size_splits=group, axis=1)
        else:
            x = tf.transpose(
                x, perm=self.get_perm_from_formats(storage_format, compute_format))
            xs = tf.split(x, num_or_size_splits=group, axis=-1)

        if transpose:
            if dilations != [1] * spatial_size:
                raise RuntimeError("Cannot set non-1 dilation for conv transpose.")
            convolved = []
            for (x, weight) in zip(xs, weight_groups):
                x_spatial_shape = [
                    x_shape[storage_format.find(d)] for d in spatial_format
                ]
                weights_shape = weights.get_shape().as_list()

                # calculate output shape
                if output_shape is None:
                    conv_output_shape = [x_shape[storage_format.find("N")]] + [
                        strides[i] * (x_spatial_shape[i] - 1) + weights_shape[i]
                        for i in list(range(spatial_size))
                    ]
                    conv_output_shape.insert(compute_c_idx, weights_shape[-2])
                else:
                    conv_output_shape = [output_shape[0]] + [
                        s + pads[i] + pads[spatial_size + i]
                        for i, s in enumerate(output_shape[2:])
                    ]
                    conv_output_shape.insert(compute_c_idx, output_shape[1])

                # make strides to match input rank
                strides_full = [1] + strides
                strides_full.insert(compute_c_idx, 1)

                # get corresponding function in tf_backed
                if spatial_size == 1:
                    conv_func = tf.contrib.nn.conv1d_transpose
                elif spatial_size == 2:
                    conv_func = tf.nn.conv2d_transpose
                elif spatial_size == 3:
                    conv_func = tf.nn.conv3d_transpose
                else:
                    raise NotImplementedError(
                        "Transposed convolution for {}d is not implemented in Tensorflow".
                            format(spatial_size))

                # use raw input x to do transposed conv
                conv_rs = conv_func(
                    x,
                    weights,
                    conv_output_shape,
                    strides_full,
                    padding="VALID",
                    data_format=compute_format)

                # pad output first by output_padding attr
                if output_padding is not None and output_shape is None:
                    output_padding = [[0, 0]
                                      ] + [[0, p] for p in output_padding]
                    output_padding.insert(compute_c_idx, [0, 0])
                    conv_rs = tf.pad(conv_rs, output_padding)

                # remove pads set in pads attr
                conv_rs_shape = conv_rs.get_shape().as_list()
                begin = [0] + pads[:spatial_size]
                begin.insert(compute_c_idx, 0)
                size = [
                    s if d in ["N", "C"] else s - pads[spatial_format.find(d)] -
                                              pads[spatial_format.find(d) + spatial_size]
                    for d, s in zip(compute_format, conv_rs_shape)
                ]
                conv_rs = tf.slice(conv_rs, begin=begin, size=size)

                convolved.append(conv_rs)
        else:
            convolved = [
                tf.nn.convolution(
                    x,
                    weight,
                    "VALID",
                    strides=strides,
                    dilation_rate=dilations,
                    data_format=compute_format)
                for (x, weight) in zip(xs, weight_groups)
            ]

        if not is_bias:
            if support_cuda:
                output = tf.concat(convolved, axis=1)
            else:
                output = tf.concat(convolved, axis=-1)
                output = tf.transpose(
                    output,
                    perm=self.get_perm_from_formats(compute_format, storage_format))
        else:
            bias = b
            bias = self._explicit_broadcast(
                bias, broadcast_dim=compute_c_idx, total_num_dim=x_rank)

            if support_cuda:
                output = tf.concat(convolved, axis=1)
                output = tf.add(output, bias)
            else:
                output = tf.concat(convolved, axis=-1)
                output = tf.add(output, bias)
                output = tf.transpose(
                    output,
                    perm=self.get_perm_from_formats(compute_format, storage_format))
        return output

    def _conv(self, X, W, has_kernel_shape,
              B=None,
              kernel_shape=None,
              dilations=None,
              strides=None,
              pads=None,
              group=None,
              output_shape=None,
              output_padding=None,
              transpose=False):
        """ Convolution method for both conv and transposed conv
        For transposed conv,
          Attr pads is not used for input, but declares how much output is padded.
          Here, output means output from transposed conv which already pad output_padding if set.
          So the pseudo explanation for output should be:
            output = conv_transpose_output + output_padding - pads
          And conv_transpose_output shape should be:
            conv_transpose_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
        """
        x = X
        x_rank = len(x.get_shape())
        x_shape = x.get_shape().as_list()
        spatial_size = x_rank - 2

        support_cuda = self.supports_device("CUDA")
        storage_format, compute_format = self.get_data_format(x_rank, support_cuda)
        compute_c_idx = compute_format.find("C")
        spatial_format = "".join([d for d in compute_format if d not in ["N", "C"]])

        in_weights = W
        weights_rank = len(in_weights.get_shape())
        if transpose:
            # Translate weights from (C x M x KH x KW) to (KH x KW X M X C)
            perm = list(range(2, weights_rank)) + [1, 0]
        else:
            # Translate weights from (M x C x KH x KW) to (KH x KW X C X M)
            perm = list(range(2, weights_rank)) + [1, 0]

        if has_kernel_shape:
            assert in_weights.get_shape().as_list()[2:] == kernel_shape, (
                "kernel_shape "
                "attr of convolution does not match the actual weight "
                "passed to this operation, attr {}, actual {}").format(
                kernel_shape,
                in_weights.get_shape().as_list())

        weights = tf.transpose(in_weights, perm)
        dilations = [1] * spatial_size if dilations is None else dilations
        strides = [1] * spatial_size if strides is None else strides

        pads = [0, 0] * spatial_size if pads is None else pads

        if not transpose:
            x = self.get_padding_as_op(x, pads)

        group = 1 if group is None else group

        weight_groups = tf.split(weights, num_or_size_splits=group, axis=-1)

        if support_cuda:
            xs = tf.split(x, num_or_size_splits=group, axis=1)
        else:
            x = tf.transpose(
                x, perm=self.get_perm_from_formats(storage_format, compute_format))
            xs = tf.split(x, num_or_size_splits=group, axis=-1)

        if transpose:
            if dilations != [1] * spatial_size:
                raise RuntimeError("Cannot set non-1 dilation for conv transpose.")
            convolved = []
            for (x, weight) in zip(xs, weight_groups):
                x_spatial_shape = [
                    x_shape[storage_format.find(d)] for d in spatial_format
                ]
                weights_shape = weights.get_shape().as_list()

                # calculate output shape
                if output_shape is None:
                    conv_output_shape = [x_shape[storage_format.find("N")]] + [
                        strides[i] * (x_spatial_shape[i] - 1) + weights_shape[i]
                        for i in list(range(spatial_size))
                    ]
                    conv_output_shape.insert(compute_c_idx, weights_shape[-2])
                else:
                    conv_output_shape = [output_shape[0]] + [
                        s + pads[i] + pads[spatial_size + i]
                        for i, s in enumerate(output_shape[2:])
                    ]
                    conv_output_shape.insert(compute_c_idx, output_shape[1])

                # make strides to match input rank
                strides_full = [1] + strides
                strides_full.insert(compute_c_idx, 1)

                # get corresponding function in tf_backed
                if spatial_size == 1:
                    conv_func = tf.contrib.nn.conv1d_transpose
                    strides_full = strides[0]
                elif spatial_size == 2:
                    conv_func = tf.nn.conv2d_transpose
                elif spatial_size == 3:
                    conv_func = tf.nn.conv3d_transpose
                else:
                    raise NotImplementedError(
                        "Transposed convolution for {}d is not implemented in Tensorflow".
                            format(spatial_size))

                # use raw input x to do transposed conv
                conv_rs = conv_func(
                    x,
                    weights,
                    conv_output_shape,
                    strides_full,
                    padding="VALID",
                    data_format=compute_format)

                # pad output first by output_padding attr
                if output_padding is not None and output_shape is None:
                    output_padding = [[0, 0]
                                      ] + [[0, p] for p in output_padding]
                    output_padding.insert(compute_c_idx, [0, 0])
                    conv_rs = tf.pad(conv_rs, output_padding)

                # remove pads set in pads attr
                conv_rs_shape = conv_rs.get_shape().as_list()
                begin = [0] + pads[:spatial_size]
                begin.insert(compute_c_idx, 0)
                size = [
                    s if d in ["N", "C"] else s - pads[spatial_format.find(d)] -
                                              pads[spatial_format.find(d) + spatial_size]
                    for d, s in zip(compute_format, conv_rs_shape)
                ]
                conv_rs = tf.slice(conv_rs, begin=begin, size=size)

                convolved.append(conv_rs)
        else:
            convolved = [
                tf.nn.convolution(
                    x,
                    weight,
                    "VALID",
                    strides=strides,
                    dilation_rate=dilations,
                    data_format=compute_format)
                for (x, weight) in zip(xs, weight_groups)
            ]

        if B is None:
            if support_cuda:
                output = tf.concat(convolved, axis=1)
            else:
                output = tf.concat(convolved, axis=-1)
                output = tf.transpose(
                    output,
                    perm=self.get_perm_from_formats(compute_format, storage_format))
        else:
            bias = B
            bias = self._explicit_broadcast(
                bias, broadcast_dim=compute_c_idx, total_num_dim=x_rank)

            if support_cuda:
                output = tf.concat(convolved, axis=1)
                output = tf.add(output, bias)
            else:
                output = tf.concat(convolved, axis=-1)
                output = tf.add(output, bias)
                output = tf.transpose(
                    output,
                    perm=self.get_perm_from_formats(compute_format, storage_format))

        return output

    def _explicit_broadcast(cls, tensor, broadcast_dim=1, total_num_dim=4):
        if broadcast_dim < 0:
            broadcast_dim += total_num_dim
        dims = [broadcast_dim + i for i in range(len(tensor.shape))]
        for i in range(total_num_dim):
            if i not in dims:
                tensor = tf.expand_dims(tensor, i)

        return tensor
        
       
    def _compatibility_pool(cls, X, kernel_shape, strides, pads, auto_pad, pooling_type, count_include_pad):

        def py_pool(x, kernel_shape, strides, pads, out_shape, count_include_pad,
                    pooling_type):
            pooling_type = pooling_type.decode('UTF-8')
            x_shape = np.shape(x)
            spatial_size = len(x_shape[2:])
            pad_attr = [(0, 0), (0, 0)] + [
                (pads[i], pads[i + spatial_size]) for i in range(spatial_size)
            ]
            constant_values = np.nan if count_include_pad == 0 else 0
            padded = np.pad(
                x, pad_attr, mode="constant", constant_values=constant_values)
            pad_shape = [
                pads[i] + pads[i + spatial_size] for i in range(spatial_size)
            ]

            y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

            for shape in itertools.product(
                    range(x_shape[0]), range(x_shape[1]), *[
                        range(
                            int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]
                                ) / strides[i] + 1)) for i in range(spatial_size)
                    ]):
                window = padded[shape[0], shape[1]]
                window_vals = np.array([
                    window[i] for i in list(
                        itertools.product(*[
                            range(strides[i] * shape[i + 2],
                                  strides[i] * shape[i + 2] + kernel_shape[i])
                            for i in range(spatial_size)
                        ]))
                ])
                if pooling_type == 'AVG':
                    f = np.average
                elif pooling_type == 'MAX':
                    f = np.max
                else:
                    raise NotImplementedError(
                        'Pooling type {} does not support. Should be AVG, MAX'.format(
                            pooling_type))

                if count_include_pad == 0:
                    y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
                else:
                    y[shape] = f(window_vals)
            return y.astype(np.float32)

        x = X
        x_shape = x.shape.as_list()
        spatial_size = len(x_shape) - 2
        kernel_shape = kernel_shape
        strides = strides
        pads = pads if pads is not None else [0] * spatial_size * 2
        auto_pad = auto_pad if auto_pad is not None else ""
        count_include_pad = count_include_pad if count_include_pad is not None else 0

        out_shape, pads = cls._pool_get_shapes(auto_pad, x_shape[2:], kernel_shape,
                                               strides, pads)

        pooled = tf.py_func(py_pool, [
            x, kernel_shape, strides, pads, out_shape, count_include_pad,
            pooling_type
        ], tf.float32)
        pooled.set_shape(x_shape[0:2] + out_shape)
        return pooled

    def visit_batchnormalization(self, op: d5.ops.BatchNormalization, network: TensorflowNetwork):
        X, B, scale, r_mean, r_var = network.fetch_internal_tensors([
            op.i_X, op.i_B, op.i_scale, op.i_mean, op.i_var])

        momentum = 0.9 if op.momentum is None else op.momentum.get_value()
        epsilon = op.epsilon.get_value() if op.epsilon else 1e-5

        # Axis is fixed to 1 since ONNX forces the NCHW data layout.
        tfop = tf.layers.BatchNormalization(axis=1, momentum=momentum,
                                            epsilon=epsilon)
        Y = tfop.apply(X, training=self.is_training)

        # Add network initializers for running mean, variance, and gamma/beta
        network.initializers[tfop.gamma] = op.i_scale
        if op.i_B is not None:
            network.initializers[tfop.beta] = op.i_B
        network.initializers[tfop.moving_mean] = op.i_mean
        network.initializers[tfop.moving_variance] = op.i_var

        network.feed_internal_tensor(op.o_Y, Y)


    PAD_TF_INCOMPATIBLE = "PAD_TF_INCOMPATIBLE"

    def pool(self, X, kernel_shape, pool_func, pooling_type,
             strides=None, pads=None, count_include_pad=None, auto_pad=None):
        x = X
        x_rank = len(x.get_shape())
        x_shape = x.get_shape().as_list()
        spatial_size = x_rank - 2

        support_cuda = self.supports_device("CUDA")
        storage_format, compute_format = self.get_data_format(x_rank, support_cuda)

        strides = [1] * spatial_size if strides is None else strides
        pads = pads if pads else None
        pad = TensorflowVisitor.PAD_TF_INCOMPATIBLE
        # from version 7
        count_include_pad = 0 if count_include_pad is None else count_include_pad

        # If padding is specified, try to recover it from explicit padding
        # specification to tf_backed padding mode:
        if pads is not None:
            pad = self._get_tf_pad(x_shape[2:], kernel_shape, strides, pads)
        else:
            # Neither pad nor auto_pad is specified, assume no padding.
            if auto_pad is None:
                pad = "VALID"
            # We consult auto_pad if pad is not specified and auto_pad
            # is available.
            else:
                if auto_pad == "SAME_UPPER":
                    pad = "SAME"
                elif auto_pad == "VALID":
                    pad = "VALID"
                elif auto_pad == "SAME_LOWER":
                    pad = TensorflowVisitor.PAD_TF_INCOMPATIBLE
                if count_include_pad == 1:
                    _, pads = self._pool_get_shapes(auto_pad, x_shape[2:],
                                                    kernel_shape, strides,
                                                    [0] * spatial_size * 2)

        if count_include_pad == 0:
            if pad is TensorflowVisitor.PAD_TF_INCOMPATIBLE:
                return self._compatibility_pool(X, kernel_shape, strides, pads, auto_pad, pooling_type,
                                                count_include_pad)
        else:
            if pads != [0] * spatial_size * 2:
                x = self.get_padding_as_op(x, pads)
            pad = "VALID"

        if support_cuda:
            pooled = pool_func(
                x,
                kernel_shape,
                padding=pad,
                strides=strides,
                data_format=compute_format)
        else:
            x = tf.transpose(
                x, perm=self.get_perm_from_formats(storage_format, compute_format))
            pooled = pool_func(
                x,
                kernel_shape,
                padding=pad,
                strides=strides,
                data_format=compute_format)
            pooled = tf.transpose(
                pooled, perm=self.get_perm_from_formats(compute_format, storage_format))

        return pooled

    def _pool_get_shapes(self, auto_pad, x_shape, kernel_shape, strides, pads):

        def _get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                           strides_spatial, output_spatial_shape):
            pad_shape = [0] * len(input_spatial_shape)
            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                for i in range(len(input_spatial_shape)):
                    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                                   input_spatial_shape[i]
            elif auto_pad in ("VALID", ""):
                pass
            return pad_shape

        def _get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                              strides_spatial):
            out_shape = [0] * len(input_spatial_shape)
            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                for i in range(len(input_spatial_shape)):
                    out_shape[i] = int(
                        np.ceil(
                            float(input_spatial_shape[i]) / float(strides_spatial[i])))
            elif auto_pad in ("VALID", ""):
                for i in range(len(input_spatial_shape)):
                    out_shape[i] = int(
                        np.ceil(
                            float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1))
                            / float(strides_spatial[i])))
            return out_shape

        spatial_size = len(x_shape)
        new_pads = pads[:]
        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            out_shape = _get_output_shape(auto_pad, x_shape, kernel_shape, strides)
            pad_shape = _get_pad_shape(auto_pad, x_shape, kernel_shape, strides,
                                       out_shape)
            for i in range(spatial_size):
                if auto_pad == "SAME_LOWER":
                    new_pads[i + spatial_size] = pad_shape[i] // 2
                    new_pads[i] = pad_shape[i] - new_pads[i + spatial_size]
                elif auto_pad == "SAME_UPPER":
                    new_pads[i] = pad_shape[i] // 2
                    new_pads[i + spatial_size] = pad_shape[i] - new_pads[i]
        elif auto_pad in ["", "VALID"]:
            pad_shape = [
                pads[i] + pads[i + spatial_size] for i in range(spatial_size)
            ]
            out_shape = _get_output_shape(auto_pad, np.add(x_shape, pad_shape),
                                          kernel_shape, strides)
        return out_shape, new_pads

    # input_shape, kernel_shape, strides are specified for
    # spatial dims only.
    def _get_tf_pad(self, input_shape, kernel_shape, strides, pads):
        assert pads is not None
        num_sp_dim = int(len(kernel_shape))

        if pads == [0] * num_sp_dim * 2:
            return "VALID"

        _, same_pads = self._pool_get_shapes("SAME_UPPER", input_shape, kernel_shape,
                                             strides, pads)
        if pads == same_pads:
            return "SAME"

        return TensorflowVisitor.PAD_TF_INCOMPATIBLE
    # End of adaptation from onnx-tensorflow
    
    def get_perm_from_formats(self, _from, _to):
        return list(map(lambda x: _from.find(x), _to))

    def get_data_format(self, x_rank, support_cuda):
        sp_dim_names = ["D", "H", "W"]
        sp_dim_lst = []
        for i in range(x_rank - 2):
            sp_dim_lst.append(sp_dim_names[-i - 1])

        sp_dim_string = "".join(reversed(sp_dim_lst))
        storage_format = "NC" + sp_dim_string

        if support_cuda:
            compute_format = "NC" + sp_dim_string
        else:
            compute_format = "N" + sp_dim_string + "C"
        return storage_format, compute_format

    def get_random_name(self):
        self.counter += 1
        return "random_name_{}".format(self.counter)

    def get_padding_as_op(self, X, pads):
        n_dim = int(len(pads) / 2)
        tf_pads = np.transpose(np.array(pads).reshape([2, n_dim]))
        tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

        padding = tf.constant(
            np.array(tf_pads).reshape([n_dim + 2, 2])
                .astype(np.int32))  # tf_backed requires int32 paddings
        return tf.pad(X, padding)

    def supports_device(self, device_name):
        return False

