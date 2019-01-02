import ctypes
import os
from typing import List, Dict, Union

import numpy as np
from jinja2 import Template

from deep500.lv0.operators.cmake_wrapper import cmake, try_mkdirs
from deep500.lv0.operators.op_compiler import CompilableOp
from deep500.utils.tensor_desc import TensorDescriptor
from deep500.lv0.operators.operator_interface import CustomOp, CustomCPPOp

_CTYPES = {
    np.int8:  'int8_t',
    np.int16: 'int16_t',
    np.int32: 'int32_t',
    np.int64: 'int64_t',
    np.uint8:  'uint8_t',
    np.uint16: 'uint16_t',
    np.uint32: 'uint32_t',
    np.uint64: 'uint64_t',
    np.float32: 'float',
    np.float64: 'double'
}


class NumpyCPPOp(CustomCPPOp):
    def convert_input(self, input: np.ndarray):
        return ctypes.c_void_p(input.__array_interface__['data'][0])


def _ctup(lst: List[TensorDescriptor], prefix: str):
    return [(_CTYPES[t.dtype], prefix+"_t"+str(i))
            for i, t in enumerate(lst)]

def desc_from_tensor(tensor: np.ndarray) -> TensorDescriptor:
    """ Converts a NumPy array to a Deep500 TensorDescriptor. """
    return TensorDescriptor(tensor.dtype.type, list(tensor.shape))


def _tendesc(lst: List[np.ndarray]):
    return [desc_from_tensor(tensor) for tensor in lst]


class OpCompiler(object):
    # Creates a wrapper file and returns its path
    def create_wrapper(self, opname: str, dirname: str,
                       input_tensors: List[TensorDescriptor], 
                       output_tensors: List[TensorDescriptor],
                       is_cuda: bool, files: List[str]):
        curpath = os.path.abspath(os.path.dirname(__file__))
        ext = ('.cpp' if not is_cuda else '.cu')

        # Read wrapper template
        template_file = os.path.join(curpath, 'base.tmpl.cpp')
        with open(template_file, 'r') as f:
            tmpl = Template(f.read())

        # Render template with tensor types
        pfile = tmpl.render(input_tensors=_ctup(input_tensors, 'inp'),
                            output_tensors=_ctup(output_tensors, 'out'),
                            nextop_grads=_ctup(output_tensors, 'nxtgrad'),
                            input_grads=_ctup(input_tensors, 'inpgrad'),
                            opfile='"' + os.path.abspath(files[0]) + '"',
                            opname=opname)

        # Try to create a directory for the wrapper file
        try_mkdirs(dirname)

        wrapper_filename = os.path.join(dirname, 'base' + ext)
        with open(wrapper_filename, 'w') as fp:
            fp.write(pfile)

        return [os.path.abspath(wrapper_filename)]

    def compile_op(self, name: str, files: List[str],
                   input_tensors: List[TensorDescriptor], 
                   output_tensors: List[TensorDescriptor],
                   is_cuda: bool,
                   live_output: bool, 
                   additional_cmake_options: List[str], 
                   additional_definitions: Dict[str, str]):
        cmakelists_path = os.path.dirname(os.path.abspath(__file__))

        # Create output folder
        dirname = '%s_%s_build' % (name, 'base')
        
        # Create wrapper template
        wrapper_files = self.create_wrapper(name, dirname, input_tensors, 
                                            output_tensors, 
                                            is_cuda, files)

        # Compile dynamic library (and ignore main file, which is part of the wrapper)
        return cmake(name, files[1:] + wrapper_files, cmakelists_path, dirname,
                     live_output=live_output, 
                     additional_cmake_options=additional_cmake_options,
                     additional_definitions=additional_definitions)

def custom_op(op_or_compilable_op: Union[CustomOp, CompilableOp]) -> CustomOp:
    """ Converts a custom operator or a compilable operator into
        an operator that can be run within the reference framework. """
    if isinstance(op_or_compilable_op, CompilableOp):
        op = op_or_compilable_op
        # Compile operator
        so_file = OpCompiler().compile_op(
            op.name, op.files, op.inputs, op.outputs,
            any([f.endswith('.cu') for f in op.files]), op.live_output, 
            op.cmake_options,
            additional_definitions=op.defs)

        return NumpyCPPOp(so_file, op.inputs,
                          op.outputs)

    elif isinstance(op_or_compilable_op, CustomOp):
        # Already a custom operator, return it as-is
        return op_or_compilable_op
    else:
        raise TypeError
