import os
import platform
from six import StringIO
import subprocess
from typing import List


# Specialized exception classes
class CompilerConfigurationError(Exception):
    pass


class CompilationError(Exception):
    pass


def try_mkdirs(dirname: str):
    try:
        os.makedirs(dirname)
    except (OSError, FileExistsError):
        pass


def _shared_lib_extension():
    """ Returns the shared library (DLL, SO) extension based on the platform. """
    osname = platform.system()
    if osname == 'Linux':
        return 'so'
    elif osname == 'Windows':
        return 'dll'
    elif osname == 'Darwin':
        return 'dylib'
    else:
        raise OSError('Unrecognized operating system')


# Command output print wrapper
def _run_liveoutput(command, print_output=False, shell=False, cwd=None):
    process = subprocess.Popen(
        command, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=shell,
        cwd=cwd)

    output = StringIO()
    while True:
        line = process.stdout.readline().rstrip()
        if not line:
            break
        if print_output:
            print(line.decode('utf-8'), flush=True)
        output.write(line.decode('utf-8'))
    stdout, _ = process.communicate()
    if print_output:
        print(stdout.decode('utf-8'), flush=True)
    output.write(stdout.decode('utf-8'))

    # An error occurred, raise exception
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command,
                                            output.getvalue())


def cmake(operator_name: str, files: List[str], cmakelists_folder: str, 
          output_folder: str, build_configuration="RelWithDebInfo", 
          live_output=False, extra_libs=[], additional_cmake_options=[],
          additional_definitions={}):
    """
        Use CMake to compile a list of files as a Deep500 custom C++ operator.
        @param operator_name The name of the operator (used in the linking
                             phase to identify class to link).
        @param files A list of files (.cpp, .cu) to compile.
        @param cmakelists_folder Chooses a framework CMakeLists.txt file.
        @param output_folder The folder in which the output is built.
        @param build_configuration The CMake build configuration to use (e.g., 
                                   Debug, RelWithDebInfo, Release)
        @param live_output Print compilation output during configuration and 
                           compilation.
        @param extra_libs A list of strings pointing to extra libraries to 
                          link the operator against.
        @param additional_cmake_options Additional flags to pass to CMake
                                        configuration.
        @param additional_definitions Additional macro definitions to pass to 
                                      the C++ preprocessor.
        @return Full path to the output shared library file.
    """

    ##############################################
    # Configure

    cmake_command = (
        ["cmake"] +
        (["-A", "x64"] if os.name == 'nt' else []) +  # Windows-specific flag
        [cmakelists_folder,
         "-DD500_FILES={}".format(";".join(files)),
         "-DD500_OPNAME={}".format(operator_name),
         "-DD500_LIBS={}".format(";".join(extra_libs))] +
        additional_cmake_options +
        ["-DEXTRA_DEFS={}".format(";".join([
            '%s=%s' % (k, v) for k, v in additional_definitions.items()]))]
    )

    # Replace backslashes with forward slashes
    cmake_command = [cmd.replace('\\', '/') for cmd in cmake_command]

    # Create output directory, if it doesn't exist
    try:
        os.makedirs(output_folder)
    except OSError:
        pass

    # Initiate CMake configuration
    try:
        _run_liveoutput(cmake_command, cwd=output_folder, 
                        print_output=live_output)
    except subprocess.CalledProcessError as ex:
        if live_output:
            raise CompilerConfigurationError('Configuration failure:\n')
        else:
            raise CompilerConfigurationError(
                'Configuration failure:\n' + ex.output)

    ##############################################
    # Compile and link
    try:
        _run_liveoutput([
            "cmake", "--build", ".", "--config", build_configuration], 
            cwd=output_folder, print_output=live_output)
    except subprocess.CalledProcessError as ex:
        # If unsuccessful, print results
        if live_output:
            raise CompilationError('Compiler failure:\n')
        else:
            raise CompilationError('Compiler failure:\n' + ex.output)

    shared_library_path = os.path.join(output_folder, "lib{}.{}".format(
        operator_name, _shared_lib_extension()))

    return shared_library_path
