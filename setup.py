from setuptools import setup, find_packages
import glob
import os

# Find C++ files by obtaining the module path and trimming the absolute path
# of the resulting files.
d500_path = os.path.dirname(os.path.abspath(__file__)) + '/deep500/'
cpp_files = [
    f[len(d500_path):]
    for f in glob.glob(d500_path + 'frameworks/reference/custom_operators/cpp/**/*', recursive=True)
]

setup(
    name='deep500',
    version='0.4.0',
    url='https://www.deep500.org/',
    author='SPCL @ ETH Zurich',
    author_email='talbn@inf.ethz.ch',
    description='The deep learning metaframework',
    packages=find_packages(),
    package_data={
        '': [
            'lv0/operators/include/deep500/*.h',
            'frameworks/*/custom_operators/CMakeLists.txt',
            'frameworks/*/custom_operators/*.cpp',
            'frameworks/caffe2/support/*',
        ] + cpp_files
    },
    include_package_data=True,
    install_requires=[
        'onnx',
        'numpy',
        'tqdm',
        'cmake',
    ],
)
