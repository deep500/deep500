from deep500.frameworks.reference.custom_operators.base import desc_from_tensor
from deep500.frameworks.reference.custom_operators.custom_op_python import *
from deep500.frameworks.reference.custom_operators.python.conv_op_common import get_output_shape as get_output_shape_conv
from deep500.frameworks.reference.custom_operators.python.pool_op_common import get_output_shape as get_output_shape_pool

from onnx import numpy_helper

import numpy as np
import deep500 as d5

from .reference_network import ReferenceNetwork


class ReferenceBuildGraphVisitor(d5.OnnxBaseVisitor):

    def __init__(self):
        self.counter = 0
        self.add_counter = 0
        self.averagepool_counter = 0
        self.batchnormalization_counter = 0
        self.conv_counter = 0
        self.gemm_counter  = 0
        self.globalaveragepool_counter = 0
        self.matmul_counter = 0
        self.maxpool_counter = 0
        self.relu_counter = 0
        self.reshape_counter = 0
        self.softmax_counter = 0
        self.sum_counter = 0
        self.squeeze_counter = 0
        self.unsqueeze_counter = 0

        self.loss_counter = 0
        self.net_input = {}

    def visit_graph(self, graph: d5.ops.OnnxGraph, network: ReferenceNetwork):
        self.net_input.clear()

    def visit_net_input(self, input: d5.ops.OnnxValueInfo, network: ReferenceNetwork):
        # TODO HBS: support multiple types
        if isinstance(input.type, d5.ops.OnnxTensorType):
            tensor_type = input.type
            self.net_input[input.name] = (tensor_type.type.to_numpy(), tensor_type.shape.shape)
        else:
            raise NotImplementedError('only tensor input supported currently')

    def visit_net_output(self, output: d5.ops.OnnxValueInfo, network: ReferenceNetwork):
        network.output_names.append(output.name)

    def visit_initializer(self, initializer: d5.ops.OnnxTensor, network: ReferenceNetwork):
        network.feed_tensor(initializer.name, initializer.data, is_param=True)

        if initializer.name in self.net_input:
            del self.net_input[initializer.name]

    def visit_constant(self, op: d5.ops.Constant, network: ReferenceNetwork):

        tensor = op.value.get_value()
        #tensor = tf.convert_to_tensor(op.value.get_value())
        network.feed_internal_tensor(op.o_output, tensor)

    '''
    def visit_initializer_end(self, network: ReferenceNetwork):
        for name, (numpy_type, shape) in self.net_input.items():
            placeholder = tf.placeholder(dtype=numpy_type, shape=shape, name=name)
            network.feed_internal_tensor(name, placeholder)
    '''

    def get_op_node_name(self, op, op_counter):  # type: (NodeProto, int)
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_counter)
        else:
            node_name = '%s (op#%d)' % (op.op_type, op_counter)

        return node_name

    def add_node(self, op, network, custom_operator, op_counter):
        op_name = self.get_op_node_name(op, op_counter)
        network.graph.add_node(
            op_name,
            operator=op,
            attribute=op.attributes,
            custom_op=custom_operator
        )
        print('added to graph: ', op_name)

    def add_edges(self, op, network, op_counter):
        op_name = self.get_op_node_name(op, op_counter)
        for input_name in op.input:
            network.tensors_op_inputs[input_name] = op_name
            if input_name in network.tensors_op_outputs:
                network.graph.add_edge(network.tensors_op_outputs[input_name], op_name)
        for output_name in op.output:
            network.tensors_op_outputs[output_name] = op_name
            if output_name in network.tensors_op_inputs:
                network.graph.add_edge(op_name, network.tensors_op_inputs[output_name])

    def visit_add(self, op: d5.ops.Add, network: ReferenceNetwork):
        A = network.variables[op.i_A]
        B = network.variables[op.i_B]
        C = np.zeros((A + B).shape, dtype=A.dtype)
        network.variables[op.o_C] = C
        network.gradients[op.i_A] = np.zeros_like(A)
        network.gradients[op.i_B] = np.zeros_like(B)
        input_desc = [desc_from_tensor(A), desc_from_tensor(B)]
        output_desc = [desc_from_tensor(C)]
        
        addlayer =  AddOp(
            input_desc,
            output_desc
        )
        self.add_counter += 1

        self.add_node(op, network, addlayer, self.add_counter)
        self.add_edges(op, network, self.add_counter)

    def visit_averagepool(self, op: d5.ops.AveragePool, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        auto_pad = 'NOTSET' if op.auto_pad is None else op.auto_pad.get_value()
        count_include_pad = 0 if op.count_include_pad is None else op.count_include_pad.get_value()
        kernel_shape = op.kernel_shape.get_value()
        temp_pads = []
        temp_strides = []
        for i in range(0, len(kernel_shape)):
            temp_pads.append(0)
            temp_pads.append(0)
            temp_strides.append(1)
        pads = temp_pads if op.pads is None else op.pads.get_value()
        strides = temp_strides if op.strides is None else op.strides.get_value()
        if auto_pad != 'NOTSET':
            out_shape = get_output_shape_pool(
                auto_pad,
                X.shape[2:],
                kernel_shape,
                strides
            )
        else:
            input_spatial_shape = X.shape[2:]
            out_shape = [0] * len(input_spatial_shape)
            for i in range(len(input_spatial_shape)):
                out_shape[i] = int(np.ceil(float(
                    input_spatial_shape[i] - (kernel_shape[i] - 1)) / float(strides[i]))
                    + pads[i] + pads[i + len(kernel_shape)]
                )
        Y_shape = np.zeros([X.shape[0], X.shape[1]] + list(out_shape), dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        input_desc = [desc_from_tensor(X)]
        output_desc = [desc_from_tensor(Y_shape)]

        averagepoollayer = AveragePoolOp(
            input_desc,
            output_desc,
            auto_pad,
            count_include_pad,
            kernel_shape,
            pads,
            strides
        )
        self.averagepool_counter += 1

        self.add_node(op, network, averagepoollayer, self.averagepool_counter)
        self.add_edges(op, network, self.averagepool_counter)

    def visit_batchnormalization(self, op: d5.ops.BatchNormalization, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        scale = network.variables[op.i_scale]
        B = network.variables[op.i_B]
        mean = network.variables[op.i_mean]
        var = network.variables[op.i_var]
        epsilon = 1e-5 if op.epsilon is None else op.epsilon.get_value()
        momentum = 0.9 if op.momentum is None else op.momentum.get_value()
        spatial = 1 if op.spatial is None else op.spatial.get_value()
        Y_shape = np.zeros(X.shape, dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        network.gradients[op.i_scale] = np.zeros_like(scale)
        network.gradients[op.i_B] = np.zeros_like(B)
        network.gradients[op.i_mean] = np.zeros_like(mean)
        network.gradients[op.i_var] = np.zeros_like(var)
        input_desc = [
            desc_from_tensor(X),
            desc_from_tensor(scale),
            desc_from_tensor(B),
            desc_from_tensor(mean),
            desc_from_tensor(var)]
        
        if not network.train_mode:
            output_desc = [desc_from_tensor(Y_shape)]

            batchnormlayer = BatchNormalizationInferenceOp(
                input_desc,
                output_desc,
                epsilon,
                momentum,
                spatial
            )
        else:
            mean_shape1 = np.zeros_like(mean)
            mean_shape2 = np.zeros_like(mean)
            var_shape1 = np.zeros_like(var)
            var_shape2 = np.zeros_like(var)
            output_desc = [
                desc_from_tensor(Y_shape),
                desc_from_tensor(mean_shape1),
                desc_from_tensor(var_shape1),
                desc_from_tensor(mean_shape2),
                desc_from_tensor(var_shape2)
            ]

            batchnormlayer = BatchNormalizationTrainingOp(
                input_desc,
                output_desc,
                epsilon,
                momentum,
                spatial
            )
        self.batchnormalization_counter += 1

        self.add_node(op, network, batchnormlayer, self.batchnormalization_counter)
        self.add_edges(op, network, self.batchnormalization_counter)

    def visit_conv(self, op: d5.ops.Conv, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        W = network.variables[op.i_W]
        if op.i_B is None:
            temp_B = np.zeros(W.shape[0], dtype=W.dtype)
            B = temp_B if op.i_B is None else network.variables[op.i_B]
            input_tensor = [X, W]
        else:
            B = network.variables[op.i_B]
            input_tensor = [X, W, B]
            network.gradients[op.i_B] = np.zeros_like(B)
        #optional input
        #temp_B = np.zeros(W.shape[0], dtype=W.dtype)
        #B = temp_B if len(op.input) < 3 else self.tensors[op.input[2]]
        auto_pad = 'NOTSET' if op.auto_pad is None else op.auto_pad.get_value()
        kernel_shape = W.shape[2:] if op.kernel_shape is None else op.kernel_shape.get_value()
        #default values if not specified
        temp_dilations = []
        temp_pads = []
        temp_strides = []
        for i in range(0, len(kernel_shape)):
            temp_dilations.append(1)
            temp_pads.append(0)
            temp_pads.append(0)
            temp_strides.append(1)
        dilations = temp_dilations if op.dilations is None else op.dilations.get_value()
        group = 1 if op.group is None else op.group.get_value()
        pads = temp_pads if op.pads is None else op.pads.get_value()
        strides = temp_strides if op.strides is None else op.strides.get_value()
        if auto_pad != 'NOTSET':
            out_shape = get_output_shape_conv(
                auto_pad,
                X.shape[2:],
                kernel_shape,
                dilations,
                strides
            )
        else:
            input_spatial_shape = X.shape[2:]
            out_shape = [0] * len(input_spatial_shape)
            for i in range(len(input_spatial_shape)):
                out_shape[i] = int(
                        np.floor(
                            float(
                                input_spatial_shape[i] + \
                                pads[i] + \
                                pads[i + len(kernel_shape)] - \
                                (dilations[i] * (kernel_shape[i] - 1) + 1)
                            ) / \
                            float(
                                strides[i]
                            )
                        ) + 1
                    )
        Y_shape = np.zeros([X.shape[0], W.shape[0]] + list(out_shape), dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        network.gradients[op.i_W] = np.zeros_like(W)
        input_tensor = [X, W] if len(op.input) == 2 else [X, W, B]
        input_desc = [desc_from_tensor(i) for i in input_tensor]
        output_desc = [desc_from_tensor(Y_shape)]

        convlayer = ConvOp(
            input_desc,
            output_desc,
            auto_pad,
            dilations,
            group,
            kernel_shape,
            pads,
            strides)
        self.conv_counter += 1

        self.add_node(op, network, convlayer, self.conv_counter)
        self.add_edges(op, network, self.conv_counter)

    def visit_gemm(self, op: d5.ops.Gemm, network: ReferenceNetwork):
        A = network.variables[op.i_A]
        B = network.variables[op.i_B]
        C = network.variables[op.i_C]
        transposeA = 0 if op.transA is None else op.transA.get_value()
        transposeB = 0 if op.transB is None else op.transB.get_value()
        alpha = 1.0 if op.alpha is None else op.alpha.get_value()
        beta = 1.0 if op.beta is None else op.beta.get_value()
        A_transposed = A.T if transposeA != 0 else A
        B_transposed = B.T if transposeB != 0 else B 
        Y_shape = np.zeros((np.dot(A_transposed, B_transposed)).shape, dtype=A.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_A] = np.zeros_like(A)
        network.gradients[op.i_B] = np.zeros_like(B)
        network.gradients[op.i_C] = np.zeros_like(C)
        input_desc = [desc_from_tensor(i) for i in [A, B, C]]
        output_desc = [desc_from_tensor(Y_shape)]

        gemmlayer = GemmOp(
            input_desc,
            output_desc,
            alpha,
            beta,
            transposeA,
            transposeB
        )
        self.gemm_counter += 1

        self.add_node(op, network, gemmlayer, self.gemm_counter)
        self.add_edges(op, network, self.gemm_counter)

    def visit_globalaveragepool(self, op: d5.ops.GlobalAveragePool, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        out_shape = [X.shape[0], X.shape[1]]
        for i in range(len(X.shape) - 2):
            out_shape = out_shape + [1]
        Y_shape = np.zeros(out_shape, dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        input_desc = [desc_from_tensor(X)]
        output_desc = [desc_from_tensor(Y_shape)]

        globalaveragepoollayer = GlobalAveragePoolOp(
            input_desc,
            output_desc
        )
        self.globalaveragepool_counter += 1

        self.add_node(op, network, globalaveragepoollayer, self.globalaveragepool_counter)
        self.add_edges(op, network, self.globalaveragepool_counter)

    def visit_matmul(self, op: d5.ops.MatMul, network: ReferenceNetwork):
        A = network.variables[op.i_A]
        B = network.variables[op.i_B]
        Y_shape = np.zeros((np.dot(A, B)).shape, dtype=A.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_A] = np.zeros_like(A)
        network.gradients[op.i_B] = np.zeros_like(B)
        input_desc = [desc_from_tensor(i) for i in [A, B]]
        output_desc = [desc_from_tensor(Y_shape)]

        matmullayer = MatMulOp(
            input_desc,
            output_desc,
        )
        self.matmul_counter += 1

        self.add_node(op, network, matmullayer, self.matmul_counter)
        self.add_edges(op, network, self.matmul_counter)


    def visit_maxpool(self, op: d5.ops.MaxPool, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        auto_pad = 'NOTSET' if op.auto_pad is None else op.auto_pad.get_value()
        kernel_shape = op.kernel_shape.get_value()
        temp_pads = []
        temp_strides = []
        for i in range(0, len(kernel_shape)):
            temp_pads.append(0)
            temp_pads.append(0)

            temp_strides.append(1)
        pads = temp_pads if op.pads is None else op.pads.get_value()
        storage_order = 0 #if op.storage_order is None else op.storage_order.get_value()
        strides = temp_strides if op.strides is None else op.strides.get_value()
        if auto_pad != 'NOTSET':
            out_shape = get_output_shape_pool(
                auto_pad,
                X.shape[2:],
                kernel_shape,
                strides
            )
        else:
            input_spatial_shape = X.shape[2:]
            out_shape = [0] * len(input_spatial_shape)
            for i in range(len(input_spatial_shape)):
                out_shape[i] = int(np.ceil(float(
                    input_spatial_shape[i] - \
                    (kernel_shape[i] - 1) + \
                    pads[i] + \
                    pads[i + len(kernel_shape)]
                ) / \
                float(
                    strides[i]
                    )
                )
                    
                )
        Y_shape = np.zeros([X.shape[0], X.shape[1]] + list(out_shape), dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        input_desc = [desc_from_tensor(X)]
        output_desc = [desc_from_tensor(Y_shape)]

        maxpoollayer = MaxPoolOp(
            input_desc,
            output_desc,
            auto_pad,
            kernel_shape,
            pads,
            storage_order,
            strides
        )
        self.maxpool_counter += 1

        self.add_node(op, network, maxpoollayer, self.maxpool_counter)
        self.add_edges(op, network, self.maxpool_counter)

    def visit_relu(self, op: d5.ops.Relu, network: ReferenceNetwork):
        X = network.variables[op.i_X]
        Y_shape = np.zeros(X.shape, dtype=X.dtype)
        network.variables[op.o_Y] = Y_shape
        network.gradients[op.i_X] = np.zeros_like(X)
        input_desc = [desc_from_tensor(X)]
        output_desc = [desc_from_tensor(Y_shape)]

        relulayer = ReluOp(
            input_desc,
            output_desc
        )
        self.relu_counter += 1

        self.add_node(op, network, relulayer, self.relu_counter)
        self.add_edges(op, network, self.relu_counter)

    def visit_reshape(self, op: d5.ops.Reshape, network: ReferenceNetwork):
        data = network.variables[op.i_data]
        shape = network.variables[op.i_shape]
        #compute output shape
        output_shape = np.ndarray(shape.shape, dtype=np.int64)
        for i in range(0, shape.size):
            if shape[i] == 0:
                output_shape[i] = data.shape[i]
            elif shape[i] == -1:
                input_size = data.size
                temp = 1
                for j in range(0, i):
                    if shape[j] == 0:
                        temp *= data.shape[j]
                    else:
                        temp *= shape[j]
                for j in range(i+1, shape.size):
                    if shape[j] == 0:
                        temp *= data.shape[j]
                    else:
                        temp *= shape[j]

                output_shape[i] = input_size / temp 
            else:
                output_shape[i] = shape[i]
        Y_shape = np.zeros(output_shape, dtype=data.dtype)
        network.variables[op.o_reshaped] = Y_shape
        network.gradients[op.i_data] = np.zeros_like(data)
        network.gradients[op.i_shape] = np.zeros_like(shape)
        input_desc = [desc_from_tensor(data)]
        output_desc = [desc_from_tensor(Y_shape)]

        reshapelayer = ReshapeOp(
            input_desc,
            output_desc
        )
        self.reshape_counter += 1

        self.add_node(op, network, reshapelayer, self.reshape_counter)
        self.add_edges(op, network, self.reshape_counter)

    def visit_softmax(self, op: d5.ops.Softmax, network: ReferenceNetwork):
        softmax_input = network.variables[op.i_input]
        axis = 1 if op.axis is None else op.axis.get_value()
        Y_shape = np.zeros(softmax_input.shape, dtype=softmax_input.dtype)
        network.variables[op.o_output] = Y_shape
        network.gradients[op.i_input] = np.zeros_like(softmax_input)
        input_desc = [desc_from_tensor(softmax_input)]
        output_desc = [desc_from_tensor(Y_shape)]

        softmaxlayer = SoftmaxOp(
            input_desc,
            output_desc,
            axis
        )
        self.softmax_counter += 1

        self.add_node(op, network, softmaxlayer, self.softmax_counter)
        self.add_edges(op, network, self.softmax_counter)

    def visit_squeeze(self, op: d5.ops.Squeeze, network: ReferenceNetwork):
        data = network.variables[op.i_data]
        axes = op.axes.get_value()
        squeezed = np.zeros(data.shape, dtype=data.dtype)
        squeezed = np.squeeze(squeezed, axis=tuple(axes))
        network.variables[op.o_squeezed] = squeezed
        network.gradiennts[op.i_data] = np.zeros_like(data)
        input_desc = [desc_from_tensor(data)]
        output_desc = [desc_from_tensor(squeezed)]

        squeezelayer = SqueezeOp(
            input_desc,
            output_desc,
            axes
        )
        self.squeeze_counter += 1

        self.add_node(op, network, squeezelayer, self.squeeze_counter)
        self.add_edges(op, network, self.squeeze_counter)

    def visit_sum(self, op: d5.ops.Sum, network: ReferenceNetwork):
        inputs = []
        for i in range(0, len(op.input)):
            inputs.append(network.variables[op.input[i]])
            network.gradients[op.input[i]] = np.zeros_like(network.variables[op.input[i]])
        Y_shape = np.copy(inputs[0])
        for i in range(1, len(inputs)):
            Y_shape = Y_shape + inputs[i]
        network.variables[op.o_sum] = Y_shape
        input_desc = [desc_from_tensor(i) for i in inputs]
        output_desc = [desc_from_tensor(Y_shape)]

        sumlayer = SumOp(
            input_desc,
            output_desc
        )
        self.sum_counter += 1

        self.add_node(op, network, sumlayer, self.sum_counter)
        self.add_edges(op, network, self.sum_counter)

    def visit_unsqueeze(self, op: d5.ops.Unsqueeze, network: ReferenceNetwork):
        data = network.variables[op.i_data]
        axes = op.axes.get_value()
        expanded = np.zeros(data.shape, dtype=data.dtype)
        for i in range(len(axes)):
            expanded = np.expand_dims(expanded, axis=axes[i])
        network.variables[op.o_expanded] = expanded
        network.gradients[op.i_data] = np.zeros_like(data)
        input_desc = [desc_from_tensor(data)]
        output_desc = [desc_from_tensor(expanded)]

        unsqueezelayer = UnsqueezeOp(
            input_desc,
            output_desc,
            axes
        )
        self.unsqueeze_counter += 1

        self.add_node(op, network, unsqueezelayer, self.unsqueeze_counter)
        self.add_edges(op, network, self.unsqueeze_counter)

    ### Loss function
    def visit_mean_squared_error(self, op: d5.ops.MeanSquaredError, network: ReferenceNetwork):
        X, target = network.variables[op.i_X], network.variables[op.i_target]
        input_desc = [desc_from_tensor(X), desc_from_tensor(target)] 
        L_shape = np.zeros(1, dtype=X.dtype)
        output_desc = [desc_from_tensor(L_shape)]
        network.variables[op.o_output] = L_shape

        mselayer = MeanSquaredErrorOp(
            input_desc,
            output_desc
        )
        self.loss_counter += 1

        op_name = 'mean_squared_error (loss_op#%d)' % (self.loss_counter)
        network.losses[op.o_output] = (mselayer, op, op_name)

    def visit_cross_entropy(self, op: d5.ops.CrossEntropy, network: ReferenceNetwork):
        X, target = network.variables[op.i_X], network.variables[op.i_target]
        input_desc = [desc_from_tensor(X), desc_from_tensor(target)] 
        L_shape = np.zeros(1, dtype=X.dtype)
        output_desc = [desc_from_tensor(L_shape)]
        network.variables[op.o_output] = L_shape

        celayer = CrossEntropyOp(
            input_desc,
            output_desc
        )
        self.loss_counter += 1

        op_name = 'cross_entropy (loss_op#%d)' % (self.loss_counter)
        network.losses[op.o_output] = (celayer, op, op_name)


    def visit_softmax_cross_entropy(self, op: d5.ops.SoftmaxCrossEntropy, network: ReferenceNetwork):
        X, target = network.variables[op.i_X], network.variables[op.i_target]
        input_desc = [desc_from_tensor(X), desc_from_tensor(target)] 
        L_shape = np.zeros(1, dtype=X.dtype)
        output_desc = [desc_from_tensor(L_shape)]
        network.variables[op.o_output] = L_shape

        lcelayer = SoftmaxCrossEntropyOp(
            input_desc,
            output_desc
        )
        self.loss_counter += 1

        op_name = 'label_cross_entropy (loss_op#%d)' % (self.loss_counter)
        network.losses[op.o_output] = (lcelayer, op, op_name)