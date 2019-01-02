#include <deque>
#include "deep500/deep500.h"
#include "../utility/tensor/copy_data.hpp"

/*
does currently not support numpy-style multi-directional broadcasting
i.e. assumes input tensors have the same shape
*/

template<typename T>
class AddLayer : public deep500::CustomOperator{

protected:
    const std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;

public: 
    //constructor
    AddLayer(std::deque<uint32_t> input_shape) :
    m_input_shape(input_shape) {
        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;
    }

    //copy constructor

    //destructor
    virtual ~AddLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, const T *input_tensor2, T *output_tensor) {
        
        /*
        ONNX notation:
        A = input
        B = input2
        C = output
        */

        for(size_t i = 0; i < m_input_size; i++) {
            output_tensor[i] = input_tensor[i] + input_tensor2[i];
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad) {

        utilityTensor::copy_data(m_input_size, nextop_grad, input_tensor_grad);

        utilityTensor::copy_data(m_input_size, nextop_grad, input_tensor2_grad);

        return;
    }


};

D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, 
                                    int num_inputs,
                                    deep500::tensor_t *param_descriptors,
                                    int num_params,
                                    deep500::tensor_t *output_descriptors,
                                    int num_outputs) {

    std::deque<uint32_t> input_tensor_shape(
        input_descriptors[0].sizes,
        input_descriptors[0].sizes + input_descriptors[0].dims
    );

    return new AddLayer<DTYPE>(input_tensor_shape);
}

D500_REGISTER_OP(AddLayer<DTYPE>);