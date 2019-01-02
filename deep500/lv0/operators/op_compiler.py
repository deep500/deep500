import ctypes
import os
import tempfile
from typing import List, Any
import numpy as np

from deep500.utils.tensor_desc import TensorDescriptor
from deep500.lv0.operators.cmake_wrapper import cmake
from deep500.lv0.operators.operator_interface import CustomCPPOp

class CompilableOp(object):
    """ A data structure representing all the necessary metadata to compile
        a Deep500 custom C++ operator. """

    def __init__(self, name, files, inputs, outputs, live_output,
                 cmake_options, defs, output_folder):
        self.name = name
        self.files = files
        self.inputs = inputs
        self.outputs = outputs
        self.live_output = live_output
        self.cmake_options = cmake_options
        self.defs = defs
        self.output_folder = output_folder

def compile_custom_cppop(operator_name: str, main_file: str,
                         input_tensors: List[TensorDescriptor],
                         output_tensors: List[TensorDescriptor],
                         other_files: List[str] = [],
                         live_output=False, additional_cmake_options=[], 
                         additional_definitions={}, 
                         output_folder='') -> CompilableOp:
    """
        Compile a list of files as a Deep500 custom C++ operator.
        @param operator_name The name of the operator (should match class in 
                             code).
        @param main_file The file (.cpp, .cu) that includes the Operator class.
        @param input_tensors List of input tensor descriptors.
        @param output_tensors List of output tensor descriptors.
        @param other_files An optional list of files (.cpp, .cu) to compile.
        @param live_output Print compilation output during configuration and 
                           compilation.
        @param additional_cmake_options Additional flags to pass to CMake
                                        configuration.
        @param additional_definitions Additional macro definitions to pass to 
                                      the C++ preprocessor.
        @param output_folder Folder to write temporary build files to
        @return A compilable operator object that can be compiled with any
                of the available frameworks.
    """
    return CompilableOp(operator_name, [main_file] + other_files,
                        input_tensors, output_tensors, live_output,
                        additional_cmake_options, additional_definitions,
                        output_folder)


def compile_custom_cppop_inline(operator_name: str, code: str,
                      input_tensors: List[Any], 
                      output_tensors: List[Any],
                      live_output=False, is_cuda=False, 
                      additional_cmake_options=[], additional_definitions={},
                      rendered_wrapper: str=None, 
                      output_folder='') -> CompilableOp:
    """
        Compile a string as a Deep500 custom C++ operator.
        @param operator_name The name of the operator (should match class in
                             code).
        @param code A code string to compile.
        @param input_tensors List of input tensor descriptors.
        @param output_tensors List of output tensor descriptors.
        @param live_output Print compilation output during configuration and
                           compilation.
        @param additional_cmake_options Additional flags to pass to CMake
                                        configuration.
        @param additional_definitions Additional macro definitions to pass to
                                      the C++ preprocessor.
        @param rendered_wrapper If not None, writes the string to a wrapper
                                file and links via CMake (used for
                                per-framework templates).
        @param output_folder Folder to write temporary build files to
        @return A compilable operator object that can be compiled with any
                of the available frameworks.
    """
    dirname = os.path.join(output_folder, '%s_build' % (operator_name))
    srcdirname = os.path.join(output_folder, '%s_src' % (operator_name))

    # Write a temporary file
    filename = os.path.abspath(
        os.path.join(srcdirname, operator_name +
                     ('.cu' if is_cuda else '.cpp')))
    try:
        os.makedirs(srcdirname)
    except OSError:
        pass
    with open(filename, 'w') as fp:
        fp.write(code)
    #######################

    files_to_compile = [os.path.abspath(filename)]

    # Add wrapper file, if necessary
    if rendered_wrapper is not None:
        wfile = os.path.abspath(os.path.join(srcdirname, framework + 
                                ('.cu' if is_cuda else '.cpp')))
        with open(wfile, 'w') as fp:
            fp.write(rendered_wrapper)
        # files_to_compile.append(wfile)
        files_to_compile = [wfile]
    #######################

    return CompilableOp(operator_name, files_to_compile,
                        input_tensors, output_tensors, live_output,
                        additional_cmake_options, additional_definitions,
                        output_folder)

