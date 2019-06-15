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

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='deep500',
    version='0.2.0',
    url='https://github.com/deep500/deep500',
    author='SPCL @ ETH Zurich',
    author_email='talbn@inf.ethz.ch',
    description='The deep learning metaframework',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
        'jinja2',
        'pillow'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
