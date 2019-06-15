import logging

from caffe2.python.modeling import initializers
from caffe2.python import core

import deep500 as d5

logging.basicConfig()
log = logging.getLogger("Caffe2Visitor")
log.setLevel(logging.INFO)

from .caffe2_network import Caffe2Network


class Caffe2Visitor(d5.OnnxBaseVisitor):
    def __init__(self, device_option):
        self.counter = 0
        self.device_option = device_option
        # if a custom op is used we dynamically load the dll
        self.custom_ops_loaded = False

    def get_random_name(self):
        self.counter += 1
        return "random_name_{}".format(self.counter)

    def visit_model(self, model: d5.ops.OnnxModel, network: Caffe2Network):
        pass

    def visit_graph(self, graph: d5.ops.OnnxGraph, network: Caffe2Network):
        # self.net: core.Net = core.Net(graph.name if graph.name is not None else self.get_random_name())
        # workspace.SwitchWorkspace(graph.name, True)
        # workspace.ResetWorkspace()
        pass

    def visit_net_output(self, output: d5.ops.OnnxValueInfo, network: Caffe2Network):
        network.output_dict[output.name] = None

    def visit_conv(self, op: d5.ops.Conv, network: Caffe2Network):
        use_bias = op.i_B is not None
        # kernel_shape, pads, strides, auto_pad, dilations, group = self.visit_conv_base(op)
        # args = self.visit_conv_base(op)
        args = self.get_conv_base(op)
        input = [op.i_X, op.i_W]
        if use_bias:
            input.append(op.i_B)

        network.train_net.Conv(input, [op.o_Y], **args)
        network.test_net.Conv(input, [op.o_Y], **args)

    def visit_pad(self, op: d5.ops.Pad, network: Caffe2Network):
        args = {}

        if op.mode is not None:
            args['mode'] = op.mode.get_value()

        pads = op.pads.get_value()
        pads[:] = pads[2:4] + pads[6:8]
        args['pads'] = pads

        if op.value is not None:
            args['value'] = op.value.get_value()

        network.train_net.PadImage([op.i_data], [op.o_output], **args)
        network.test_net.PadImage([op.i_data], [op.o_output], **args)

    def visit_mean(self, op: d5.ops.Mean, network: Caffe2Network):
        network.train_net.Mean(op.input, [op.o_mean])
        network.test_net.Mean(op.input, [op.o_mean])

    def visit_min(self, op: d5.ops.Min, network: Caffe2Network):
        network.train_net.Min(op.input, [op.o_min])
        network.test_net.Min(op.input, [op.o_min])

    def visit_max(self, op: d5.ops.Max, network: Caffe2Network):
        network.train_net.Max(op.input, [op.o_max])
        network.test_net.Max(op.input, [op.o_max])

    def visit_mul(self, op: d5.ops.Mul, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Mul([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Mul([op.i_A, op.i_B], [op.o_C], **args)

    def visit_or(self, op: d5.ops.Or, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Or([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Or([op.i_A, op.i_B], [op.o_C], **args)

    def visit_not(self, op: d5.ops.Not, network: Caffe2Network):
        network.train_net.Not([op.i_X], [op.o_Y])
        network.test_net.Not([op.i_X], [op.o_Y])

    def visit_neg(self, op: d5.ops.Neg, network: Caffe2Network):
        network.train_net.Negative([op.i_X], [op.o_Y])
        network.test_net.Negative([op.i_X], [op.o_Y])

    def visit_pow(self, op: d5.ops.Pow, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Pow([op.i_X, op.i_Y], [op.o_Z], **args)
        network.test_net.Pow([op.i_X, op.i_Y], [op.o_Z], **args)

    def visit_softplus(self, op: d5.ops.Softplus, network: Caffe2Network):
        network.train_net.Softplus([op.i_X], [op.o_Y])
        network.test_net.Softplus([op.i_X], [op.o_Y])

    def visit_softsign(self, op: d5.ops.Softsign, network: Caffe2Network):
        network.train_net.Softsign([op.i_input], [op.o_output])
        network.test_net.Softsign([op.i_input], [op.o_output])

    def visit_sigmoid(self, op: d5.ops.Sigmoid, network: Caffe2Network):
        network.train_net.Sigmoid([op.i_X], [op.o_Y])
        network.test_net.Sigmoid([op.i_X], [op.o_Y])

    def visit_tanh(self, op: d5.ops.Tanh, network: Caffe2Network):
        network.train_net.Tanh([op.i_input], [op.o_output])
        network.test_net.Tanh([op.i_input], [op.o_output])

    def visit_split(self, op: d5.ops.Split, network: Caffe2Network):
        network.train_net.Split([op.i_input], [op.output], axis=op.axis, split=op.split, order='NCHW')
        network.test_net.Split([op.i_input], [op.output], axis=op.axis, split=op.split, order='NCHW')

    def visit_sqrt(self, op: d5.ops.Sqrt, network: Caffe2Network):
        network.train_net.Sqrt([op.i_X], [op.o_Y])
        network.test_net.Sqrt([op.i_X], [op.o_Y])

    def visit_sub(self, op: d5.ops.Sub, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Sub([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Sub([op.i_A, op.i_B], [op.o_C], **args)

    def visit_elu(self, op: d5.ops.Elu, network: Caffe2Network):
        network.train_net.Elu([op.i_X], [op.o_Y], alpha=op.alpha.get_value())
        network.test_net.Elu([op.i_X], [op.o_Y], alpha=op.alpha.get_value())

    def visit_flatten(self, op: d5.ops.Flatten, network: Caffe2Network):
        axis = 1 if op.axis is None else op.axis.get_value()
        network.train_net.Flatten([op.i_input], [op.o_output], axis=axis)
        network.test_net.Flatten([op.i_input], [op.o_output], axis=axis)

    def visit_exp(self, op: d5.ops.Exp, network: Caffe2Network):
        network.train_net.Exp([op.i_input], [op.o_output])
        network.test_net.Exp([op.i_input], [op.o_output])

    def visit_equal(self, op: d5.ops.Equal, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.EQ([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.EQ([op.i_A, op.i_B], [op.o_C], **args)

    def visit_xor(self, op: d5.ops.Xor, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Xor([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Xor([op.i_A, op.i_B], [op.o_C], **args)

    def visit_unsqueeze(self, op: d5.ops.Unsqueeze, network: Caffe2Network):
        dims = op.axes.get_value()
        network.train_net.ExpandDims([op.i_data], [op.o_expanded], dims=dims)
        network.test_net.ExpandDims([op.i_data], [op.o_expanded], dims=dims)

    def visit_squeeze(self, op: d5.ops.Squeeze, network: Caffe2Network):
        dims = op.axes.get_value()
        network.train_net.Squeeze([op.i_data], [op.o_squeezed], dims=dims)
        network.test_net.Squeeze([op.i_data], [op.o_squeezed], dims=dims)

    def visit_transpose(self, op: d5.ops.Transpose, network: Caffe2Network):
        args = {}
        if op.perm is not None:
            args['axes'] = op.perm.get_value()

        network.train_net.Transpose([op.i_data], [op.o_transposed], **args)
        network.test_net.Transpose([op.i_data], [op.o_transposed], **args)

    def visit_topk(self, op: d5.ops.TopK, network: Caffe2Network):
        k = op.k.get_value()
        network.train_net.TopK([op.i_X], [op.o_Values, op.o_Indices, self.get_random_name()], k=k)
        network.test_net.TopK([op.i_X], [op.o_Values, op.o_Indices, self.get_random_name()], k=k)

    def visit_thresholdedrelu(self, op: d5.ops.ThresholdedRelu, network: Caffe2Network):
        alpha = 1.0 if op.alpha is None else op.alpha.get_value()
        network.train_net.ThresholdedRelu([op.i_X], [op.o_Y], alpha=alpha)
        network.test_net.ThresholdedRelu([op.i_X], [op.o_Y], alpha=alpha)

    def visit_selu(self, op: d5.ops.Selu, network: Caffe2Network):
        alpha = 1.6732 if op.alpha is None else op.alpha.get_value()
        scale = 1.0507 if op.gamma is None else op.gamma.get_value()
        network.train_net.Selu([op.i_X], [op.o_Y], alpha=alpha, scale=scale)
        network.test_net.Selu([op.i_X], [op.o_Y], alpha=alpha, scale=scale)

    def visit_leakyrelu(self, op: d5.ops.LeakyRelu, network: Caffe2Network):
        alpha = 0.01 if op.alpha is None else op.alpha.get_value()
        network.train_net.LeakyRelu([op.i_X], [op.o_Y], alpha=alpha)
        network.test_net.LeakyRelu([op.i_X], [op.o_Y], alpha=alpha)

    def visit_size(self, op: d5.ops.Size, network: Caffe2Network):
        network.train_net.Size([op.i_data], [op.o_size])
        network.test_net.Size([op.i_data], [op.o_size])

    def visit_shape(self, op: d5.ops.Shape, network: Caffe2Network):
        network.train_net.Shape([op.i_data], [op.o_shape])
        network.test_net.Shape([op.i_data], [op.o_shape])

    def visit_greater(self, op: d5.ops.Greater, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.GE([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.GE([op.i_A, op.i_B], [op.o_C], **args)

    def visit_less(self, op: d5.ops.Less, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.LE([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.LE([op.i_A, op.i_B], [op.o_C], **args)

    def visit_log(self, op: d5.ops.Log, network: Caffe2Network):
        network.train_net.Log([op.i_input], [op.o_output])
        network.test_net.Log([op.i_input], [op.o_output])

    def visit_floor(self, op: d5.ops.Floor, network: Caffe2Network):
        network.train_net.Floor([op.i_X], [op.o_Y])
        network.test_net.Floor([op.i_X], [op.o_Y])

    def visit_if(self, op: d5.ops.If, network: Caffe2Network):
        args = {}
        args['then_net'] = op.then_branch.get_value()
        if op.else_branch is not None:
            args['else_net'] = op.else_branch.get_value()
        network.train_net.If([op.i_cond], **args)
        network.test_net.If([op.i_cond], **args)

    def visit_instancenormalization(self, op: d5.ops.InstanceNormalization, network: Caffe2Network):
        network.train_net.InstanceNorm([op.i_input, op.i_B, op.i_scale],
                                       [op.o_output], epsilon=op.epsilon, order='NCHW')
        network.test_net.InstanceNorm([op.i_input, op.i_B, op.i_scale],
                                      [op.o_output], epsilon=op.epsilon, order='NCHW')

    def visit_upsample(self, op: d5.ops.Upsample, network: Caffe2Network):
        network.train_net.ResizeNearest([op.i_X], [op.o_Y],
                                        height_scale=op.height_scale.get_value(),
                                        width_scale=op.width_scale.get_value())
        network.test_net.ResizeNearest([op.i_X], [op.o_Y],
                                       height_scale=op.height_scale.get_value(),
                                       width_scale=op.width_scale.get_value())

    def visit_maxpool(self, op: d5.ops.MaxPool, network: Caffe2Network):
        args = self.get_conv_base(op)

        network.train_net.MaxPool([op.i_X], [op.o_Y], **args)
        network.test_net.MaxPool([op.i_X], [op.o_Y], **args)

    def visit_globalmaxpool(self, op: d5.ops.GlobalMaxPool, network: Caffe2Network):
        args = self.get_conv_base(op)
        args['global_pooling'] = 1
        network.train_net.MaxPool([op.i_X], [op.o_Y], **args)
        network.test_net.MaxPool([op.i_X], [op.o_Y], **args)

    def visit_add(self, op: d5.ops.Add, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Add([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Add([op.i_A, op.i_B], [op.o_C], **args)

    def visit_ceil(self, op: d5.ops.Ceil, network: Caffe2Network):
        network.train_net.Ceil([op.i_X], [op.o_Y])
        network.test_net.Ceil([op.i_X], [op.o_Y])

    def visit_clip(self, op: d5.ops.Clip, network: Caffe2Network):
        args = {}
        if op.min is not None:
            args['min'] = op.min.get_value()
        if op.max is not None:
            args['max'] = op.max.get_value()
        network.train_net.Clip([op.i_input], [op.o_output], **args)
        network.test_net.Clip([op.i_input], [op.o_output], **args)

    def visit_relu(self, op: d5.ops.Relu, network: Caffe2Network):
        network.train_net.Relu([op.i_X], [op.o_Y])
        network.test_net.Relu([op.i_X], [op.o_Y])

    def visit_concat(self, op: d5.ops.Concat, network: Caffe2Network):
        network.train_net.Concat(op.input, [op.o_concat_result, self.get_random_name()], axis=op.axis.get_value())
        network.test_net.Concat(op.input, [op.o_concat_result, self.get_random_name()], axis=op.axis.get_value())

    def visit_constant(self, op: d5.ops.Constant, network: Caffe2Network):
        # network.init_net.ConstantFill([], [op.o_output], value=op.value.get_value().get_data())
        network.workspace.FeedBlob(op.o_output, op.value.get_value(), self.device_option)

    def visit_convtranspose(self, op: d5.ops.ConvTranspose, network: Caffe2Network):
        args = self.visit_conv_base(op)
        input = [op.i_X, op.i_W] if op.i_B is None else [op.i_X, op.i_W, op.i_B]
        network.train_net.ConvTranspose(input, [op.o_Y], **args)
        network.test_net.ConvTranspose(input, [op.o_Y], **args)

    def visit_matmul(self, op: d5.ops.MatMul, network: Caffe2Network):
        network.train_net.BatchMatMul([op.i_A, op.i_B], [op.o_Y])
        network.test_net.BatchMatMul([op.i_A, op.i_B], [op.o_Y])

    def visit_reshape(self, op: d5.ops.Reshape, network: Caffe2Network):
        network.train_net.Reshape(op.input, [op.o_reshaped, self.get_random_name()])
        network.test_net.Reshape(op.input, [op.o_reshaped, self.get_random_name()])

    def get_simple_math_op(self, op):
        args = {}
        # if op.axis is not None:
        #    args['axis'] = op.axis.get_value()
        args['broadcast'] = 1  # if op.broadcast is None else op.broadcast.get_value()
        return args

    def visit_div(self, op: d5.ops.Div, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.Div([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.Div([op.i_A, op.i_B], [op.o_C], **args)

    def visit_abs(self, op: d5.ops.Abs, network: Caffe2Network):
        network.train_net.Abs([op.i_X], [op.o_Y])
        network.test_net.Abs([op.i_X], [op.o_Y])

    def visit_and(self, op: d5.ops.And, network: Caffe2Network):
        args = self.get_simple_math_op(op)
        network.train_net.And([op.i_A, op.i_B], [op.o_C], **args)
        network.test_net.And([op.i_A, op.i_B], [op.o_C], **args)

    def visit_dropout(self, op: d5.ops.Dropout, network: Caffe2Network):
        ratio = 0.5 if op.ratio is None else op.ratio.get_value()
        network.train_net.Dropout([op.i_data], [op.o_output, op.o_mask], is_test=0, ratio=ratio)
        network.test_net.Dropout([op.i_data], [op.o_output], is_test=1, ratio=ratio)

    def visit_batchnormalization(self, op: d5.ops.BatchNormalization, network: Caffe2Network):
        args = {}
        if op.epsilon is not None:
            args['epsilon'] = op.epsilon.get_value()
        if op.momentum is not None:
            args['momentum'] = op.momentum.get_value()
        if op.spatial is not None:
            args['spatial'] = op.spatial.get_value()

        output = [op.o_Y]
        input = [op.i_X, op.i_scale, op.i_B, op.i_mean, op.i_var]

        args['is_test'] = 1
        network.test_net.SpatialBN(input, output, **args)

        args['is_test'] = 0
        op.o_mean = op.o_mean if op.o_mean is not None else op.i_mean
        op.o_var = op.o_var if op.o_var is not None else op.i_var
        op.o_saved_mean = op.o_saved_mean if op.o_saved_mean is not None else self.get_random_name()
        op.o_saved_var = op.o_saved_var if op.o_saved_var is not None else self.get_random_name()

        output.extend([op.o_mean, op.o_var, op.o_saved_mean, op.o_saved_var])
        network.train_net.SpatialBN(input, output, **args)

    def visit_initializer(self, initializer: d5.ops.OnnxTensor, network: Caffe2Network):
         with core.DeviceScope(self.device_option):
            network.train_init_net.AddExternalInput(initializer.name)
            network.test_net.AddExternalInput(initializer.name)
            network.train_model.create_param(initializer.name, initializer.dims,
                                             initializer=initializers.ExternalInitializer())
            network.test_model.create_param(initializer.name, initializer.dims,
                                            initializer=initializers.ExternalInitializer())
            network.workspace.FeedBlob(initializer.name, initializer.get_data(), self.device_option)

    def visit_sum(self, op: d5.ops.Sum, network: Caffe2Network):
        network.train_net.Sum(op.input, [op.o_sum])
        network.test_net.Sum(op.input, [op.o_sum])

    def visit_averagepool(self, op: d5.ops.AveragePool, network: Caffe2Network):
        args = self.get_conv_base(op)
        network.train_net.AveragePool([op.i_X], [op.o_Y], **args)
        network.test_net.AveragePool([op.i_X], [op.o_Y], **args)

    def visit_globalaveragepool(self, op: d5.ops.GlobalAveragePool, network: Caffe2Network):
        args = self.get_conv_base(op)
        args['global_pooling'] = 1
        network.train_net.AveragePool([op.i_X], [op.o_Y], **args)
        network.test_net.AveragePool([op.i_X], [op.o_Y], **args)

    def visit_gemm(self, op: d5.ops.Gemm, network: Caffe2Network):
        (A, B, C) = op.i_A, op.i_B, op.i_C
        alpha = 1.0 if op.alpha is None else op.alpha.get_value()
        beta = 1.0 if op.beta is None else op.beta.get_value()
        trans_a = 0 if op.transA is None else op.transA.get_value()
        trans_b = 0 if op.transB is None else op.transB.get_value()
        broadcast = 1  # if op.broadcast is None else op.broadcast.get_value()

        if alpha != 1:
            A = self.get_random_name()
            network.train_net.Scale([op.i_A], [A], scale=alpha)
            network.test_net.Scale([op.i_A], [A], scale=alpha)
        if beta != 0:
            C = self.get_random_name()
            network.train_net.Scale([op.i_C], [C], scale=beta)
            network.test_net.Scale([op.i_C], [C], scale=beta)

        if not trans_a and trans_b and broadcast:
            network.train_net.FC([A, B, C], [op.o_Y])
            network.test_net.FC([A, B, C], [op.o_Y])
        else:
            AB = self.get_random_name()
            network.train_net.MatMul([A, B], [AB], trans_a=trans_a, trans_b=trans_b)
            network.test_net.MatMul([A, B], [AB], trans_a=trans_a, trans_b=trans_b)
            network.train_net.Add([AB, C], [op.o_Y], broadcast=broadcast)
            network.test_net.Add([AB, C], [op.o_Y], broadcast=broadcast)

    def visit_softmax(self, op: d5.ops.Softmax, network: Caffe2Network):
        axis = 1 if op.axis is None else op.axis.get_value()
        network.train_net.Softmax([op.i_input], [op.o_output], axis=axis)
        network.test_net.Softmax([op.i_input], [op.o_output], axis=axis)

    # TODO(HBS): Clean up these 2 conv base methods, just keep one!
    def get_conv_base(self, op):
        args = {}
        if hasattr(op, 'kernel_shape') and op.kernel_shape is not None:
            args['kernels'] = op.kernel_shape.get_value()
        if hasattr(op, 'pads') and op.pads is not None:
            args['pads'] = op.pads.get_value()
            if 'kernels' in args and len(args['kernels']) == len(args['pads']):
                args['pads'] = args['pads'] * 2
        if hasattr(op, 'strides') and op.strides is not None:
            args['strides'] = op.strides.get_value()
        if hasattr(op, 'autopad') and op.auto_pad is not None:
            args['autopad'] = op.auto_pad.get_value()
        return args

    def visit_conv_base(self, op):
        args = {}

        if hasattr(op, 'kernel_shape'):
            args['kernels'] = op.kernel_shape.get_value()
        if hasattr(op, 'strides'):
            args['strides'] = [1, 1] if op.strides is None else op.strides.get_value()
        if hasattr(op, 'auto_pad') and op.auto_pad is not None:
            args['auto_pad'] = op.auto_pad.get_value()
        elif hasattr(op, 'pads'):
            pads = [0, 0] if op.pads is None else op.pads.get_value()
            if 'kernels' in args:
                args['pads'] = pads * 2 if len(pads) == len(args['kernels']) else pads
            else:
                args['pads'] = pads

        if hasattr(op, 'group'):
            args['group'] = 1 if op.group is None else op.group.get_value()

        if hasattr(op, 'dilations'):
            args['dilations'] = [1, 1] if op.dilations is None else op.dilations.get_value()

        return args

    def visit_cross_entropy(self, cross_entroy: d5.ops.CrossEntropy, network: Caffe2Network):
        random_name = self.get_random_name()

        network.train_net.CrossEntropy(cross_entroy.input, random_name)
        network.test_net.CrossEntropy(cross_entroy.input, random_name)

        network.train_net.AveragedLoss(random_name, cross_entroy.output)
        network.test_net.AveragedLoss(random_name, cross_entroy.output)

        # network.train_model.AddGradientOperators([cross_entroy.output])
        network.output_dict[cross_entroy.output] = None

    def visit_softmax_cross_entropy(self, label_cross_entropy: d5.ops.SoftmaxCrossEntropy, network: Caffe2Network):
        random_name = self.get_random_name()

        network.train_net.SoftmaxCrossEntropy(label_cross_entropy.input, random_name)
        network.test_net.SoftmaxCrossEntropy(label_cross_entropy.input, random_name)

        network.train_net.AveragedLoss(random_name, label_cross_entropy.output)
        network.test_net.AveragedLoss(random_name, label_cross_entropy.output)

        # network.train_model.AddGradientOperators([label_cross_entropy.output])
        network.output_dict[label_cross_entropy.output] = None

    def visit_mean_squared_error(self, op: d5.ops.MeanSquaredError, network: Caffe2Network):
        random_name = self.get_random_name()

        network.train_net.SquaredL2Distance(op.input, random_name)
        network.test_net.SquaredL2Distance(op.input, random_name)

        network.train_net.AveragedLoss(random_name, op.output)
        network.test_net.AveragedLoss(random_name, op.output)

        # network.train_model.AddGradientOperators([op.output])
        network.output_dict[op.output] = None

    def visit_stopgradient(self, stop_gradient: d5.ops.StopGradient, network: Caffe2Network):
        network.train_net.StopGradient([stop_gradient.i_input], [stop_gradient.o_output])
        network.test_net.StopGradient([stop_gradient.i_input], [stop_gradient.o_output])
