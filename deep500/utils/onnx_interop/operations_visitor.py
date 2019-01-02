import abc
from deep500.lv1.network import Network
from deep500.utils.onnx_interop.generated_operators import *


class OperationsVisitor(abc.ABC):
    def visit_sin(self, op: Sin, network: Network):
        raise Exception('implement this method')

    def visit_atan(self, op: Atan, network: Network):
        raise Exception('implement this method')

    def visit_asin(self, op: Asin, network: Network):
        raise Exception('implement this method')

    def visit_acos(self, op: Acos, network: Network):
        raise Exception('implement this method')

    def visit_unsqueeze(self, op: Unsqueeze, network: Network):
        raise Exception('implement this method')

    def visit_topk(self, op: TopK, network: Network):
        raise Exception('implement this method')

    def visit_tile(self, op: Tile, network: Network):
        raise Exception('implement this method')

    def visit_thresholdedrelu(self, op: ThresholdedRelu, network: Network):
        raise Exception('implement this method')

    def visit_tanh(self, op: Tanh, network: Network):
        raise Exception('implement this method')

    def visit_sum(self, op: Sum, network: Network):
        raise Exception('implement this method')

    def visit_squeeze(self, op: Squeeze, network: Network):
        raise Exception('implement this method')

    def visit_spacetodepth(self, op: SpaceToDepth, network: Network):
        raise Exception('implement this method')

    def visit_softmax(self, op: Softmax, network: Network):
        raise Exception('implement this method')

    def visit_slice(self, op: Slice, network: Network):
        raise Exception('implement this method')

    def visit_size(self, op: Size, network: Network):
        raise Exception('implement this method')

    def visit_shape(self, op: Shape, network: Network):
        raise Exception('implement this method')

    def visit_selu(self, op: Selu, network: Network):
        raise Exception('implement this method')

    def visit_transpose(self, op: Transpose, network: Network):
        raise Exception('implement this method')

    def visit_scaledtanh(self, op: ScaledTanh, network: Network):
        raise Exception('implement this method')

    def visit_sigmoid(self, op: Sigmoid, network: Network):
        raise Exception('implement this method')

    def visit_scale(self, op: Scale, network: Network):
        raise Exception('implement this method')

    def visit_reducesumsquare(self, op: ReduceSumSquare, network: Network):
        raise Exception('implement this method')

    def visit_reducesum(self, op: ReduceSum, network: Network):
        raise Exception('implement this method')

    def visit_reshape(self, op: Reshape, network: Network):
        raise Exception('implement this method')

    def visit_reduceprod(self, op: ReduceProd, network: Network):
        raise Exception('implement this method')

    def visit_tan(self, op: Tan, network: Network):
        raise Exception('implement this method')

    def visit_globalaveragepool(self, op: GlobalAveragePool, network: Network):
        raise Exception('implement this method')

    def visit_reducel2(self, op: ReduceL2, network: Network):
        raise Exception('implement this method')

    def visit_meanvariancenormalization(self, op: MeanVarianceNormalization, network: Network):
        raise Exception('implement this method')

    def visit_gru(self, op: GRU, network: Network):
        raise Exception('implement this method')

    def visit_giventensorfill(self, op: GivenTensorFill, network: Network):
        raise Exception('implement this method')

    def visit_multinomial(self, op: Multinomial, network: Network):
        raise Exception('implement this method')

    def visit_flatten(self, op: Flatten, network: Network):
        raise Exception('implement this method')

    def visit_exp(self, op: Exp, network: Network):
        raise Exception('implement this method')

    def visit_equal(self, op: Equal, network: Network):
        raise Exception('implement this method')

    def visit_not(self, op: Not, network: Network):
        raise Exception('implement this method')

    def visit_sqrt(self, op: Sqrt, network: Network):
        raise Exception('implement this method')

    def visit_elu(self, op: Elu, network: Network):
        raise Exception('implement this method')

    def visit_reducemin(self, op: ReduceMin, network: Network):
        raise Exception('implement this method')

    def visit_div(self, op: Div, network: Network):
        raise Exception('implement this method')

    def visit_prelu(self, op: PRelu, network: Network):
        raise Exception('implement this method')

    def visit_depthtospace(self, op: DepthToSpace, network: Network):
        raise Exception('implement this method')

    def visit_gruunit(self, op: GRUUnit, network: Network):
        raise Exception('implement this method')

    def visit_convtranspose(self, op: ConvTranspose, network: Network):
        raise Exception('implement this method')

    def visit_logsoftmax(self, op: LogSoftmax, network: Network):
        raise Exception('implement this method')

    def visit_reducelogsum(self, op: ReduceLogSum, network: Network):
        raise Exception('implement this method')

    def visit_reducemean(self, op: ReduceMean, network: Network):
        raise Exception('implement this method')

    def visit_crop(self, op: Crop, network: Network):
        raise Exception('implement this method')

    def visit_and(self, op: And, network: Network):
        raise Exception('implement this method')

    def visit_reducemax(self, op: ReduceMax, network: Network):
        raise Exception('implement this method')

    def visit_argmax(self, op: ArgMax, network: Network):
        raise Exception('implement this method')

    def visit_lpnormalization(self, op: LpNormalization, network: Network):
        raise Exception('implement this method')

    def visit_loop(self, op: Loop, network: Network):
        raise Exception('implement this method')

    def visit_affine(self, op: Affine, network: Network):
        raise Exception('implement this method')

    def visit_lstm(self, op: LSTM, network: Network):
        raise Exception('implement this method')

    def visit_softplus(self, op: Softplus, network: Network):
        raise Exception('implement this method')

    def visit_randomnormallike(self, op: RandomNormalLike, network: Network):
        raise Exception('implement this method')

    def visit_argmin(self, op: ArgMin, network: Network):
        raise Exception('implement this method')

    def visit_conv(self, op: Conv, network: Network):
        raise Exception('implement this method')

    def visit_add(self, op: Add, network: Network):
        raise Exception('implement this method')

    def visit_abs(self, op: Abs, network: Network):
        raise Exception('implement this method')

    def visit_split(self, op: Split, network: Network):
        raise Exception('implement this method')

    def visit_batchnormalization(self, op: BatchNormalization, network: Network):
        raise Exception('implement this method')

    def visit_upsample(self, op: Upsample, network: Network):
        raise Exception('implement this method')

    def visit_globallppool(self, op: GlobalLpPool, network: Network):
        raise Exception('implement this method')

    def visit_reducelogsumexp(self, op: ReduceLogSumExp, network: Network):
        raise Exception('implement this method')

    def visit_matmul(self, op: MatMul, network: Network):
        raise Exception('implement this method')

    def visit_sub(self, op: Sub, network: Network):
        raise Exception('implement this method')

    def visit_maxpool(self, op: MaxPool, network: Network):
        raise Exception('implement this method')

    def visit_neg(self, op: Neg, network: Network):
        raise Exception('implement this method')

    def visit_xor(self, op: Xor, network: Network):
        raise Exception('implement this method')

    def visit_greater(self, op: Greater, network: Network):
        raise Exception('implement this method')

    def visit_dropout(self, op: Dropout, network: Network):
        raise Exception('implement this method')

    def visit_cast(self, op: Cast, network: Network):
        raise Exception('implement this method')

    def visit_gather(self, op: Gather, network: Network):
        raise Exception('implement this method')

    def visit_ceil(self, op: Ceil, network: Network):
        raise Exception('implement this method')

    def visit_concat(self, op: Concat, network: Network):
        raise Exception('implement this method')

    def visit_softsign(self, op: Softsign, network: Network):
        raise Exception('implement this method')

    def visit_constantfill(self, op: ConstantFill, network: Network):
        raise Exception('implement this method')

    def visit_hardmax(self, op: Hardmax, network: Network):
        raise Exception('implement this method')

    def visit_identity(self, op: Identity, network: Network):
        raise Exception('implement this method')

    def visit_if(self, op: If, network: Network):
        raise Exception('implement this method')

    def visit_imagescaler(self, op: ImageScaler, network: Network):
        raise Exception('implement this method')

    def visit_randomuniform(self, op: RandomUniform, network: Network):
        raise Exception('implement this method')

    def visit_cos(self, op: Cos, network: Network):
        raise Exception('implement this method')

    def visit_gemm(self, op: Gemm, network: Network):
        raise Exception('implement this method')

    def visit_instancenormalization(self, op: InstanceNormalization, network: Network):
        raise Exception('implement this method')

    def visit_relu(self, op: Relu, network: Network):
        raise Exception('implement this method')

    def visit_averagepool(self, op: AveragePool, network: Network):
        raise Exception('implement this method')

    def visit_less(self, op: Less, network: Network):
        raise Exception('implement this method')

    def visit_log(self, op: Log, network: Network):
        raise Exception('implement this method')

    def visit_loopindextensor(self, op: LoopIndexTensor, network: Network):
        raise Exception('implement this method')

    def visit_floor(self, op: Floor, network: Network):
        raise Exception('implement this method')

    def visit_min(self, op: Min, network: Network):
        raise Exception('implement this method')

    def visit_rnn(self, op: RNN, network: Network):
        raise Exception('implement this method')

    def visit_lppool(self, op: LpPool, network: Network):
        raise Exception('implement this method')

    def visit_max(self, op: Max, network: Network):
        raise Exception('implement this method')

    def visit_maxroipool(self, op: MaxRoiPool, network: Network):
        raise Exception('implement this method')

    def visit_lrn(self, op: LRN, network: Network):
        raise Exception('implement this method')

    def visit_mean(self, op: Mean, network: Network):
        raise Exception('implement this method')

    def visit_mul(self, op: Mul, network: Network):
        raise Exception('implement this method')

    def visit_globalmaxpool(self, op: GlobalMaxPool, network: Network):
        raise Exception('implement this method')

    def visit_pad(self, op: Pad, network: Network):
        raise Exception('implement this method')

    def visit_or(self, op: Or, network: Network):
        raise Exception('implement this method')

    def visit_reducel1(self, op: ReduceL1, network: Network):
        raise Exception('implement this method')

    def visit_parametricsoftplus(self, op: ParametricSoftplus, network: Network):
        raise Exception('implement this method')

    def visit_pow(self, op: Pow, network: Network):
        raise Exception('implement this method')

    def visit_aten(self, op: ATen, network: Network):
        raise Exception('implement this method')

    def visit_constant(self, op: Constant, network: Network):
        raise Exception('implement this method')

    def visit_randomnormal(self, op: RandomNormal, network: Network):
        raise Exception('implement this method')

    def visit_hardsigmoid(self, op: HardSigmoid, network: Network):
        raise Exception('implement this method')

    def visit_clip(self, op: Clip, network: Network):
        raise Exception('implement this method')

    def visit_randomuniformlike(self, op: RandomUniformLike, network: Network):
        raise Exception('implement this method')

    def visit_leakyrelu(self, op: LeakyRelu, network: Network):
        raise Exception('implement this method')

    def visit_reciprocal(self, op: Reciprocal, network: Network):
        raise Exception('implement this method')

