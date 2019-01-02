import ctypes
from typing import List
from deep500.utils.tensor_desc import TensorDescriptor, tensor_t


class CustomOp(object):
    """ A class that represents a callable custom operator that is not tied to
        any specific framework. Can be implemented in Python or as a shared
        object. """

    def __init__(self, input_descriptors: List[TensorDescriptor],
                 output_descriptors: List[TensorDescriptor]):
        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

    @property
    def num_inputs(self):
        return len(self._input_desc)

    @property
    def num_outputs(self):
        return len(self._output_desc)

    @property
    def input_descriptors(self) -> List[TensorDescriptor]:
        return self._input_desc

    @property
    def output_descriptors(self) -> List[TensorDescriptor]:
        return self._output_desc


    def forward(self, *inputs):
        raise NotImplementedError('Abstract class')

    def backward(self, grads, fwd_inputs, fwd_outputs):
        raise NotImplementedError('Abstract class')


class CustomPythonOp(CustomOp):
    pass


def _to_c_array(desclist: List[TensorDescriptor]):
    return (tensor_t*len(desclist))(*[desc.c_tensor for desc in desclist])


class CustomCPPOp(CustomOp):
    def __init__(self, so_file: str, input_descriptors: List[TensorDescriptor],
                 output_descriptors: List[TensorDescriptor]):
        self._lib = ctypes.CDLL(so_file)
        if not getattr(self._lib, 'create_new_op', False):
            raise ValueError('Invalid custom operator library file')

        self._input_desc = input_descriptors
        self._output_desc = output_descriptors

        # Obtain handle to operator object
        self._lib.create_new_op.restype = ctypes.c_void_p
        self._ophandle = ctypes.c_void_p(self._lib.create_new_op(
            _to_c_array(input_descriptors), len(input_descriptors),
            _to_c_array(output_descriptors), len(output_descriptors)))

        self._lib.is_cuda_supported.restype = ctypes.c_bool
        self._lib.report.restype = ctypes.c_int64

    def convert_input(self, input):
        return ctypes.c_void_p(input)

    def forward(self, *args):
        self._lib.forward_op(self._ophandle, *(self.convert_input(a)
                                               for a in args))

    def forward_cuda(self, *args):
        self._lib.forward_opCuda(self._ophandle, 
                                  *(self.convert_input(a) for a in args))

    def backward(self, *args):
        self._lib.backward_op(self._ophandle, 
                              *(self.convert_input(a) for a in args))

    def backward_cuda(self, *args):
        self._lib.backward_opCuda(self._ophandle, 
                                   *(self.convert_input(a) for a in args))

    @property
    def supports_cuda(self) -> bool:
        return self._lib.is_cuda_supported(self._ophandle)

    def report(self, data=ctypes.c_void_p(None)) -> int:
        return self._lib.report(self._ophandle, data)
