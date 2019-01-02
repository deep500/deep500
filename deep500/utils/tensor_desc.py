import ctypes
import numpy as np
from typing import List

# See C version in include/deep500/deep500.h
class tensor_t(ctypes.Structure):
    _fields_ = [("type",  ctypes.c_int),
                ("order", ctypes.c_int),
                ("dims",  ctypes.c_uint8),
                ("sizes", ctypes.POINTER(ctypes.c_uint32))]


# A tensor descriptor object
class TensorDescriptor(object):
    _ORDER = {None: 0, "NCHW": 1, "NHWC": 2}
    _TYPES = {
        np.int8:  1, # TT_INT8
        np.int16: 2, # TT_INT16
        np.int32: 3, # TT_INT32
        np.int64: 4, # TT_INT64
        np.uint8:  5, # TT_UINT8
        np.uint16: 6, # TT_UINT16
        np.uint32: 7, # TT_UINT32
        np.uint64: 8, # TT_UINT64
        #np.float16: 9, # TT_HALF
        np.float32: 10, # TT_FLOAT
        np.float64: 11, # TT_DOUBLE
    }
    def __init__(self, dtype: type, shape: List[int], order="N/A"):
        c_type = 0
        if dtype in TensorDescriptor._TYPES:
            c_type = TensorDescriptor._TYPES[dtype]
        c_order = 0
        if order in TensorDescriptor._ORDER:
            c_order = TensorDescriptor._ORDER[order]
        
        self._shape_arr = (ctypes.c_uint32 * len(shape))()
        self._shape_arr[:] = list(shape)

        self.dtype = dtype
        self.shape = shape
    
        self.c_tensor = tensor_t(c_type, c_order, len(shape), self._shape_arr)

    @staticmethod
    def runtime_shape(dtype):
        return TensorDescriptor(dtype, [])
        
