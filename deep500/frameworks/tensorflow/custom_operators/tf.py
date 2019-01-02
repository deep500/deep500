import ctypes
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from typing import List, Union, Type, Dict

from jinja2 import Template

from deep500.frameworks.reference.custom_operators.base import OpCompiler

from deep500.utils.tensor_desc import TensorDescriptor
from deep500.lv0.operators.operator_interface import CustomPythonOp, CustomOp, _to_c_array
from deep500.lv0.operators.op_compiler import CompilableOp
from deep500.lv0.operators.cmake_wrapper import cmake

def desc_from_tensor(tensor: tf.Tensor) -> TensorDescriptor:
    """ Converts a Tensorflow tensor to a Deep500 TensorDescriptor. """
    return TensorDescriptor(tensor.dtype.as_numpy_dtype, tensor.shape, "NHWC")

def custom_op_from_native(op_class: Type, inputs: List[TensorDescriptor],
                          outputs: List[TensorDescriptor]) -> CustomOp:
    """ Converts a Tensorflow operation to a Deep500 CustomOp. 
        @param op_class The Tensorflow operator class type to convert.
        @param inputs A list of tensor descriptors for the inputs.
        @param outputs A list of tensor descriptors for the outputs.
        @return A custom operator capable of returning numpy arrays as per
                the definition of CustomOp.
    """
    class CustomTFOp(CustomOp):
        def __init__(self):
            super().__init__(inputs, outputs)
        def forward(self, *inputs):
            tf_inputs = tuple(tf.placeholder(dtype=it.dtype, shape=it.shape) for it in inputs)
            return op_class(*tf_inputs).eval({iten: iarr for iten, iarr in zip(tf_inputs, inputs)})
        def backward(self, grads, fwd_inputs, fwd_outputs):
            tf_inputs = tuple(tf.placeholder(dtype=it.dtype, shape=it.shape) for it in inputs)
            feed_dict = {iten: iarr for iten, iarr in zip(tf_inputs, fwd_inputs)}
            grad_out = tf.gradients(op_class(*tf_inputs), tf_inputs, grads)
            return [o.eval(feed_dict) for o in grad_out]
    return CustomTFOp()

_DTYPE_TO_CSTR = {
    tf.float32: 'float',
    tf.float64: 'double',
    tf.float16: 'half',
    tf.uint8: 'unsigned char',
    tf.int8: 'char',
    tf.int16: 'short',
    tf.int32: 'int32',
    tf.uint32: 'int',
    tf.int64: 'long long',
    tf.uint64: 'unsigned long long'
}
def _ctup(lst: List[TensorDescriptor], prefix: str):
    return [(_DTYPE_TO_CSTR[t.dtype], prefix+"_t"+str(i)) 
            for i,t in enumerate(lst)]

class TFCompiler(OpCompiler):
    # Creates a wrapper file and returns its path
    def create_wrapper(self, opname: str, dirname: str,
                       input_tensors: List[TensorDescriptor], 
                       output_tensors: List[TensorDescriptor],
                       is_cuda: bool, files: List[str]):
        curpath = os.path.abspath(os.path.dirname(__file__))
        ext = ('.cpp' if not is_cuda else '.cu')

        # Read wrapper template
        template_file = os.path.join(curpath, 'tf.tmpl.cpp')
        with open(template_file, 'r') as f:
            tmpl = Template(f.read())

        # Render template with tensor types
        pfile = tmpl.render(input_tensors=_ctup(input_tensors, 'inp'), 
            output_tensors=_ctup(output_tensors, 'out'),
            output_shapes=[s.shape for s in output_tensors],
            nextop_grads=_ctup(output_tensors, 'nxtgrad'),
            input_grads=_ctup(input_tensors, 'inpgrad'),
            input_shapes=[s.shape for s in input_tensors],
            platforms=([('DEVICE_CPU', ''), ('DEVICE_GPU', 'Cuda')] if is_cuda else [('DEVICE_CPU', '')]),
            opfile='"' + os.path.abspath(files[0]) + '"',
            opname=opname)

        # Try to create a directory for the wrapper file
        try:
            os.makedirs(dirname)
        except (OSError, FileExistsError):
            pass

        wrapper_filename = os.path.join(dirname, 'tf' + ext)
        with open(wrapper_filename, 'w') as fp:
            fp.write(pfile)

        return [os.path.abspath(wrapper_filename)]

    def compile_op(self, name: str, files: List[str],
                   input_tensors: List[TensorDescriptor], 
                   output_tensors: List[TensorDescriptor],
                   is_cuda: bool,
                   live_output: bool, 
                   additional_cmake_options: List[str], 
                   additional_definitions: Dict[str, str],
                   output_folder: str):
        cmakelists_path = os.path.dirname(os.path.abspath(__file__))

        # Create output folder
        dirname = os.path.join(output_folder, '%s_%s_build' % (name, 'tf'))
    
        # CUDA-specific macro
        defs = {}
        defs.update(additional_definitions)
        if is_cuda:
            defs.update({'__D500_OPHASCUDA': 1})
        print('iscuda', is_cuda)
    
        # Create wrapper template
        wrapper_files = self.create_wrapper(name, dirname, input_tensors, 
                                            output_tensors, 
                                            is_cuda, files)

        # Compile dynamic library (and ignore main file, which is part of the wrapper)
        return cmake(name, files[1:] + wrapper_files, cmakelists_path, dirname,
                     live_output=live_output, 
                     additional_cmake_options=additional_cmake_options,
                     additional_definitions=defs)


class TFCompiledOp(object):
    def __init__(self, op, op_func, op_grad_func, lib):
        self.op = op
        self.op_func = op_func
        self.op_grad_func = op_grad_func
        self.lib = lib

def _custom_cpp_op(op: CompilableOp, stateful, name):
    """ Compiles and registers a custom C++ Tensorflow operator """
    # Compile the .so file
    tf_path = os.path.abspath(os.path.dirname(tf.__file__))
    
    so_file = TFCompiler().compile_op(op.name, op.files, 
        op.inputs, op.outputs,
        any([f.endswith('.cu') for f in op.files]), op.live_output,  
        additional_cmake_options=['-DTENSORFLOW_PATH=' + tf_path] + op.cmake_options,
        additional_definitions=op.defs, output_folder=op.output_folder)

    # Load the compiled library into Tensorflow
    op_module = tf.load_op_library(so_file)
    op_func = getattr(op_module, 'tf_op' + op.name)
    op_grad_func = getattr(op_module, 'tf_op_grad' + op.name)
    
    # Create the deep500 custom op object
    lib = ctypes.CDLL(so_file)
    if not getattr(lib, 'create_new_op', False):
        raise ValueError('Invalid custom operator library file')
    lib.create_new_op.restype = ctypes.c_int64
    lib.is_cuda_supported.restype = ctypes.c_bool
    lib.report.restype = ctypes.c_int64

    return TFCompiledOp(op, op_func, op_grad_func, lib)


def _create_op_handle(compiled_op):
    op = compiled_op.op
    handle = int(compiled_op.lib.create_new_op(
        _to_c_array(op.inputs), len(op.inputs),
        _to_c_array(op.outputs), len(op.outputs)))
    
    # Forward
    def op_functor(*args, **kwargs):
        return compiled_op.op_func(*args, op_handle_ptr=handle, **kwargs)
    
    # Backward
    def op_grad_functor(tfop, *args, **kwargs):
        return compiled_op.op_grad_func(*(args + tuple(tfop.inputs) + tuple(tfop.outputs)), op_handle_ptr=handle, **kwargs)
    try:
        tf.RegisterGradient('TfOp' + op.name)(op_grad_functor)
    except KeyError as ex:
        print("Warning: Gradient already registered to another handle")

    return op_functor, compiled_op.lib, handle

def custom_op(op: Union[CustomOp, CompilableOp, TFCompiledOp], stateful=True, name=None,
              use_autodiff=False, compile_only=False, return_handle=False):
    """
        Registers a custom Tensorflow operator from `CustomOp`, 
        `CompilableOp`, or `TFCompiledOp` objects.
        @param op The custom operator. If numpy is not used, automatic 
                    differentiation via Tensorflow applies.
        @param stateful True if the operation is not a pure function (enables
                        sub-expression elimination optimizations if False).
        @param name Specify a custom name for this operation.
        @param use_autodiff If true, uses tensorflow tensors, otherwise 
                            assumes numpy arrays.
        @param compile_only If true, returns a TFCompiledOp instead of an instantiated op
        @param return_handle (for C++ ops) If true, also returns a direct handle
                             to the operator object and library as a 3-tuple:
                             (operator, library, handle).
        @return A tf.Operation object (or a function) that calls the custom operator.
    """
    if isinstance(op, CompilableOp):
        result = _custom_cpp_op(op, stateful, name)
        if compile_only:
            return result
        else:
            op = result
    if isinstance(op, TFCompiledOp):
        result = _create_op_handle(op)
        if return_handle:
            return result
        else:
            return result[0]
    elif isinstance(op, CustomOp):
        if use_autodiff == True:
            return op.forward

        def _fwd(*inputs):
            return op.forward(*inputs)
        def _bwd(tfop, *grads):
            def _actual_bwd(*args):
                return op.backward(args[:len(grads)], 
                                     args[len(grads):(len(grads)+len(tfop.inputs))], 
                                     args[(len(grads)+len(tfop.inputs)):])
            return tf.py_func(_actual_bwd, 
                              (list(grads) + list(tfop.inputs) + list(tfop.outputs)), 
                              [inp.dtype for inp in op.input_descriptors], 
                              stateful=stateful)

        # Gradient replacement adapted from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342

        # Generate a unique name to avoid duplicates
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
        tf.RegisterGradient(rnd_name)(_bwd)

        def result(*inputs):
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
                return tf.py_func(_fwd, inputs, 
                                  [out.dtype for out in op.output_descriptors],
                                  stateful=stateful, name=name)
        return result

