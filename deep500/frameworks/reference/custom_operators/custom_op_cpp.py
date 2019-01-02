import os

def _import_cpp_operator(op_name):
    curpath = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(curpath, 'cpp', 'operators', op_name + '.cpp')
    return fname
    #with open(fname, 'r') as fp:
    #    return fp.read()

AddCPPOp = _import_cpp_operator('add_op')
AveragePoolCPPOp = _import_cpp_operator('averagepool_op')
BatchNormalizationInferenceCPPOp = _import_cpp_operator('batchnormalization_op_inference')
ConvCPPOp = _import_cpp_operator('conv_op')
GemmCPPOp = _import_cpp_operator('gemm_op')
GlobalAveragePoolCPPOp = _import_cpp_operator('globalaveragepool_op')
MatMulCPPOp = _import_cpp_operator('matmul_op')
MaxPoolCPPOp = _import_cpp_operator('maxpool_op')
ReluCPPOp = _import_cpp_operator('relu_op')
ReshapeCPPOp = _import_cpp_operator('reshape_op')
SoftmaxCPPOp = _import_cpp_operator('softmax_op')
#SqueezeCPPOp = _import_cpp_operator('squeeze_op')
SumCPPOp = _import_cpp_operator('sum_op')
#UnsqueezeCPPOp = _import_cpp_operator('unsqueeze_op')
