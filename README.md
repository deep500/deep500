Deep500: A Deep Learning Meta-Framework and HPC Benchmarking Library
====================================================================

<p align="center">
	<img src="d500.png" /><br />
    (or: 500 ways to train deep neural networks)
</p>


Deep500 is a library that can be used to customize and measure anything with deep neural networks, using a clean, high-performant, and simple interface. Deep500 includes four levels of abstraction: (L0) Operators (layers); (L1) Network Evaluation; (L2) Training; and (L3) Distributed Training.

Using Deep500, you automatically gain:
* Operator validation, including gradient checking for backpropagation
* Statistically-accurate performance benchmarks and plots
* High-performance integration with popular deep learning frameworks (see Supported Frameworks below)
* Running your operator/framework/optimizer/communicator/... with real workloads, alongside existing environments
* and much more...

## Installation

Using pip: `pip install git+https://github.com/deep500/deep500.git`

## Usage

See the [tutorials](tutorials/README.md).

## Requirements
 * Python 3.5 or later
 * Protobuf (`sudo apt-get install protobuf-compiler libprotoc-dev`)
 * For plotted metrics: matplotlib
 * For distributed optimization:
     * Any MPI implementation (OpenMPI, MPICH, MVAPICH etc.)
     * mpi4py Python package

## Supported Frameworks
 * Tensorflow
 * Pytorch
 * Caffe2

## Contributing

Deep500 is an open-source, community driven project. We are happy to accept Pull Requests with your contributions!
 
## License

Deep500 is published under the New BSD license, see [LICENSE](LICENSE).
