Deep500: A Deep Learning Meta-Framework and HPC Benchmarking Library
====================================================================

![Deep500](deep500.svg)
<br />
(or: 500 ways to train deep neural networks)


Deep500 is a library that can be used to customize and measure anything with deep neural networks, using a clean, high-performant, and simple interface. Deep500 includes four levels of abstraction: (L0) Operators (layers); (L1) Network Evaluation; (L2) Training; and (L3) Distributed Training.

Using Deep500, you automatically gain:
* Operator validation, including gradient checking for backpropagation
* Statistically-accurate performance benchmarks and plots
* High-performance integration with popular deep learning frameworks (see Supported Frameworks below)
* Running your operator/framework/optimizer/communicator/... with real workloads, alongside existing environments
* and much more...

## Installation

Using pip: `pip install deep500`

## Usage

See the [tutorials](https://github.com/deep500/deep500/tree/master/tutorials).

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

## Reference

If you use this meta-framework please cite it as:
```bibtex
@inproceedings{deep500,
  author={T. Ben-Nun and M. Besta and S. Huber and A. N. Ziogas and D. Peter and T. Hoefler},
  title={{A Modular Benchmarking Infrastructure for High-Performance and Reproducible Deep Learning}},
  year={2019},
  month={May},
  publisher={IEEE},
  note={The 33rd IEEE International Parallel \& Distributed Processing Symposium (IPDPS'19)},
}
```

## Contributing

Deep500 is an open-source, community driven project. We are happy to accept Pull Requests with your contributions!
 
## License

Deep500 is published under the New BSD license, see [LICENSE](https://github.com/deep500/deep500/blob/master/LICENSE).
