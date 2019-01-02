from typing import List, Dict, Union, Type
import os

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as cppext

import numpy as np
from jinja2 import Template

from deep500.frameworks.reference.custom_operators.base import OpCompiler

from deep500.lv0.operators.cmake_wrapper import cmake, try_mkdirs
from deep500.lv0.operators.operator_interface import CustomPythonOp, CustomCPPOp, CustomOp
from deep500.lv0.operators.op_compiler import CompilableOp
from deep500.utils.tensor_desc import TensorDescriptor

def _create_custom_autograd_function_tensors(pyop: CustomPythonOp):
    class CustomPytorchTensorFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            outputs = pyop.forward(args)
            if isinstance(outputs, (list, tuple)):
                variables = list(outputs) + list(args)
            else:
                variables = [outputs] + list(args)

            ctx.save_for_backward(*variables)
            return outputs

        @staticmethod
        def backward(ctx, *grads):
            bwd_outputs = pyop.backward(grads,
                                        ctx.saved_variables[pyop.num_outputs:],
                                        ctx.saved_variables[:pyop.num_outputs])
            return bwd_outputs
    return CustomPytorchTensorFunction


def _create_custom_autograd_function_py(pyop: CustomPythonOp):
    class CustomPytorchFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            dev = args[0].device
            outputs = pyop.forward(*(arg.detach().cpu().numpy() for arg in args))
            if isinstance(outputs, np.ndarray):
                outputs = [outputs]

            variables = [torch.from_numpy(v).to(device=dev) for v in outputs] + list(args)
            ctx.save_for_backward(*variables)

            return tuple(variables[:pyop.num_outputs])

        @staticmethod
        def backward(ctx, *grads):
            dev = grads[0].device
            fwd_output_tensors = ctx.saved_variables[:pyop.num_outputs]
            saved_vars = [var.detach().cpu().numpy() for var in ctx.saved_variables]
            bwd_outputs = pyop.backward(tuple(grad.cpu().numpy() for grad in grads), 
                                        saved_vars[pyop.num_outputs:],
                                        saved_vars[:pyop.num_outputs])

            # One output, convert to tuple
            if isinstance(bwd_outputs, np.ndarray): 
                bwd_outputs = (bwd_outputs,)
            bwd_outputs = tuple(torch.from_numpy(out).to(device=dev) for out in bwd_outputs)

            return bwd_outputs
    return CustomPytorchFunction

def _create_custom_autograd_function_cpp(cppop: CustomCPPOp):
    class CustomPytorchFunction(torch.autograd.Function):
        @staticmethod
        def is_cuda_supported():
            return cppop.supports_cuda

        @staticmethod
        def forward(ctx, *args):
            dev = args[0].device
            if (not getattr(cppop, '_pytorch_out', False) or cppop._pytorch_out[0].device != dev):
                cppop._pytorch_out = tuple(
                    torch.zeros(size=t.shape, dtype=_NUMPY_TO_DTYPE[t.dtype], device=dev) 
                    for t in cppop._output_desc)

            if dev.type == 'cuda':
                cppop.forward_cuda(*(arg.contiguous()._cdata for arg in (args + cppop._pytorch_out)))
            else:
                cppop.forward(*(arg.contiguous()._cdata for arg in (args + cppop._pytorch_out)))
            ctx.save_for_backward(*(args + cppop._pytorch_out))
            return cppop._pytorch_out

        @staticmethod
        def backward(ctx, *grads):
            dev = grads[0].device
            if (not getattr(cppop, '_pytorch_ingrad', False) or cppop._pytorch_ingrad[0].device != dev):
                cppop._pytorch_ingrad = [
                    torch.zeros(size=t.shape, dtype=_NUMPY_TO_DTYPE[t.dtype], device=dev) 
                    for t in cppop._input_desc]
                cppop._pytorch_ingrad_cdata = [t.contiguous()._cdata for t in cppop._pytorch_ingrad]
            
            args = [arg.contiguous()._cdata for arg in (
                grads +               # nextop_grads
                ctx.saved_variables)] # fwd_inputs, fwd_outputs

            if dev.type == 'cuda':
                if len(cppop._pytorch_ingrad) > 0:
                    cppop.backward_cuda(*(args + cppop._pytorch_ingrad_cdata))
            else:
                if len(cppop._pytorch_ingrad) > 0:
                    cppop.backward(*(args + cppop._pytorch_ingrad_cdata))

            return tuple(cppop._pytorch_ingrad)
    return CustomPytorchFunction


class CustomPytorchModule(torch.nn.Module):
    def __init__(self, func: type):
        super(CustomPytorchModule, self).__init__()
        self.op = func

    def forward(self, *args):
        return self.op.apply(*args)

class CustomPytorchCPPModule(torch.nn.Module):
    def __init__(self, func: type):
        super(CustomPytorchCPPModule, self).__init__()
        self.op = func

    def is_cuda_supported(self):
        return self.op.is_cuda_supported()

    def forward(self, *args):
        return self.op.apply(*args)

class CustomPytorchAutodiffModule(torch.nn.Module):
    def __init__(self, op: CustomPythonOp):
        super(CustomPytorchAutodiffModule, self).__init__()
        self.op = op

    def forward(self, *args):
        return self.op.forward(*args)


def _register_pytorch_pyop(pyop: CustomPythonOp, use_autodiff=False):
    """
        Registers a custom pytorch operator from a `CustomPythonOp`.
        @param pyop The custom operator. If the `backward` function is not 
                    implemented and numpy is not used, automatic 
                    differentiation via pytorch applies.
        @param use_autodiff If true, uses Pytorch tensors to automatically 
                            differentiate operator, otherwise 
                            assumes numpy arrays.
        @return A torch.nn.Module object that calls the function on demand.
    """

    # Python operator has no backward function, use autograd
    if type(pyop).backward == CustomPythonOp.backward:
        if use_autodiff == False:
            raise TypeError('For autodiff pytorch operators, numpy must not '
                            'be used')
        return CustomPytorchAutodiffModule(pyop)
    
    if use_autodiff:
        func = _create_custom_autograd_function_tensors(pyop)
    else:
        func = _create_custom_autograd_function_py(pyop)
    
    return CustomPytorchModule(func)


########### C++

_NUMPY_TO_DTYPE = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.float16: torch.float16,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64
}

_DTYPE_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64
}

_DTYPE_TO_CSTR = {
    np.float32: 'Float',
    np.float64: 'Double',
    np.float16: 'Half',
    np.uint8: 'Byte',
    np.int8: 'Char',
    np.int16: 'Short',
    np.int32: 'Int',
    np.int64: 'Long'
}

def _tendesc(lst: List[torch.Tensor]):
    return [desc_from_tensor(tensor) for tensor in lst]



def _ctup(lst: List[TensorDescriptor], prefix: str):
    return [(_DTYPE_TO_CSTR[t.dtype], prefix+"_t"+str(i)) 
            for i,t in enumerate(lst)]

class PytorchCompiler(OpCompiler):
    # Creates a wrapper file and returns its path
    def create_wrapper(self, opname: str, dirname: str,
                       input_tensors: List[TensorDescriptor], 
                       output_tensors: List[TensorDescriptor],
                       is_cuda: bool, files: List[str]):
        curpath = os.path.abspath(os.path.dirname(__file__))
        ext = ('.cpp' if not is_cuda else '.cu')

        # Read wrapper template
        template_file = os.path.join(curpath, 'pytorch.tmpl.cpp')
        with open(template_file, 'r') as f:
            tmpl = Template(f.read())

        # Render template with tensor types
        pfile = tmpl.render(input_tensors=_ctup(input_tensors, 'inp'), 
            output_tensors=_ctup(output_tensors, 'out'),
            nextop_grads=_ctup(output_tensors, 'nxtgrad'),
            input_grads=_ctup(input_tensors, 'inpgrad'),
            opfile='"' + os.path.abspath(files[0]) + '"',
            platforms=['', 'Cuda'] if is_cuda else [''])

        # Try to create a directory for the wrapper file
        try:
            os.makedirs(dirname)
        except (OSError, FileExistsError):
            pass

        wrapper_filename = os.path.join(dirname, 'pytorch' + ext)
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
        dirname = '%s_%s_build' % (name, 'pytorch')
        
        # Create wrapper template
        wrapper_files = self.create_wrapper(name, dirname, input_tensors, 
                                            output_tensors, 
                                            is_cuda, files)

        # CUDA-specific macro
        defs = {}
        defs.update(additional_definitions)
        if is_cuda:
            defs.update({'__D500_OPHASCUDA': 1})

        # Compile dynamic library (and ignore main file, which is part of the wrapper)
        return cmake(name, files[1:] + wrapper_files, cmakelists_path, dirname,
                     live_output=live_output, 
                     additional_cmake_options=additional_cmake_options,
                     additional_definitions=defs)

def _compile_pytorch_cppop(op: CompilableOp):
    torch_path = os.path.abspath(os.path.dirname(torch.__file__))


    # Convert numpy arrays to tensor descriptors and compile
    so_file = PytorchCompiler().compile_op(op.name, op.files, 
        op.inputs, op.outputs,
        any([f.endswith('.cu') for f in op.files]), op.live_output,  
        additional_cmake_options=['-DPYTORCH_PATH=' + torch_path] + op.cmake_options,
        additional_definitions=op.defs)

    cppop = CustomCPPOp(so_file, op.inputs, op.outputs)

    func = _create_custom_autograd_function_cpp(cppop)

    return CustomPytorchCPPModule(func)

########### Public API

def desc_from_tensor(tensor: torch.Tensor) -> TensorDescriptor:
    """ Converts a Pytorch tensor to a Deep500 TensorDescriptor. """
    return TensorDescriptor(_DTYPE_TO_NUMPY[tensor.dtype], list(tensor.shape))

def custom_op(op: Union[CustomOp, CompilableOp], 
              use_autodiff=False) -> torch.nn.Module:
    """ Converts a custom operator or a compilable operator into
        an operator that can be run within Pytorch as a Module.
        @param op The custom operator, or one that can be compiled.
        @param use_autodiff If true, uses Pytorch tensors to automatically
                            differentiate operator, otherwise assumes numpy 
                            arrays.
    """
    if isinstance(op, CompilableOp):
        return _compile_pytorch_cppop(op)
    elif isinstance(op, CustomCPPOp):
        raise TypeError('Pre-compiled C++ operators are not supported')
    elif isinstance(op, CustomOp):
        return _register_pytorch_pyop(op, use_autodiff)
    else:
        raise TypeError

def custom_op_from_native(op_class: Type, inputs: List[TensorDescriptor],
                          outputs: List[TensorDescriptor]) -> CustomOp:
    """ Converts a Pytorch module to a Deep500 CustomOp. 
        @param op_class The Pytorch operator/module class type to convert.
        @param inputs A list of tensor descriptors for the inputs.
        @param outputs A list of tensor descriptors for the outputs.
        @return A custom operator capable of returning numpy arrays as per
                the definition of CustomOp.
    """
    class CustomPytorchOp(CustomOp):
        def __init__(self):
            super().__init__(inputs, outputs)
        def forward(self, *input_tuple):
            pt_inputs = tuple(torch.tensor(inp, dtype=_NUMPY_TO_DTYPE[it.dtype]) for it, inp in zip(inputs, input_tuple))
            op_or_tensor = op_class(*pt_inputs)
            if isinstance(op_or_tensor, torch.Tensor):
                return op_or_tensor.numpy()
            else:
                return op_or_tensor.forward(*pt_inputs).numpy()
        def backward(self, grads, fwd_inputs, fwd_outputs):
            pt_inputs = tuple(torch.tensor(inp, requires_grad=True, dtype=_NUMPY_TO_DTYPE[it.dtype]) for inp, it in zip(fwd_inputs, inputs))
            pt_grads = tuple(torch.tensor(g, dtype=_NUMPY_TO_DTYPE[g.dtype.type]) for g in grads)
            op_or_tensor = op_class(*pt_inputs)
            if isinstance(op_or_tensor, torch.Tensor):
                pt_outputs = op_or_tensor
            else:
                pt_outputs = op_or_tensor.forward(*pt_inputs)
            input_grads = torch.autograd.grad(pt_outputs, pt_inputs, pt_grads)
            return [o.numpy() for o in input_grads]
    return CustomPytorchOp()

