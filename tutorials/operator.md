## Level 0: Custom Operator

You have a new implementation of an operator. With the meta-framework, checking for quality and performance is as easy as writing the operator itself. In this example, we use TensorFlow as the underlying framework, and implement a new operator as follows:

```python
import deep500 as d5
from deep500.frameworks import tensorflow as d5tf

# Python implementation
class MyAmazingLayer (d5.CustomPythonOp):
    def __init__(self, *args, **kwargs):
        # Optional initialization here...
        
    def forward(self, inputs):
        # Do amazing stuff
    def backward(self, grads, fwd_inputs, fwd_outputs):
        # Backpropagate amazing stuff
        
tfop = d5tf.custom_op(MyAmazingLayer)
        
# you can now use tfop as a Tensorflow layer normally!
```

If you have a fast, C++ version of your operator, you can similarly register it using a simple API:
```cpp
#include <deep500/deep500.h>
#include <cmath>

template<typename T>
class amazinglayer : public deep500::CustomOperator {
protected:
    // ...
public:
    amazinglayer(/* ... */) {}
    virtual ~amazinglayer() {}
    virtual bool supports_cuda() { return true; }

    void forward(const T *input, T *output) {
        // Stuff
    }
    void forward_cuda(const T *gpu_input, T *gpu_output) {
        // GPU stuff
    }

    void backward(const T *nextop_grad,
                  const T *fwd_input_tensor,
                  const T *fwd_output_tensor,
                  T *input_tensor_grad) {
        // Backward stuff
    }
    void backward_cuda(const T *gpu_nextop_grad,
                  const T *gpu_fwd_input_tensor,
                  const T *gpu_fwd_output_tensor,
                  T *gpu_input_tensor_grad) {
        // GPU backward stuff
    }    
};
D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, 
                                  int num_inputs,
                                  deep500::tensor_t *output_descriptors,
                                  int num_outputs) {
    return new amazinglayer<DTYPE>(/* ... */);
}

D500_REGISTER_OP(amazinglayer<DTYPE>);
```

and in Python:
```python
    # Compiles the code and creates the deep500.CustomCPPOp object
    cppop = d5.compile_custom_cppop('amazinglayer', 'amazinglayer.cpp', 
        [d5.tensordesc(100, d5.float32)], [d5.tensordesc(100, d5.float32)], 
        additional_definitions={'DTYPE': 'float'})
        
    # Creates an operator that can run within the Tensorflow graph (on the CPU or GPU)
    cpptfop = d5tf.custom_op(cppop)
    
    # That's it!
```
