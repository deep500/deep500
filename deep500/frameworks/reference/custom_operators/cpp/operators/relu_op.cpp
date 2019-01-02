#include <deque>
#include "deep500/deep500.h"
#include "../utility/tensor/copy_data.hpp"

template<typename T>
class ReluLayer : public deep500::CustomOperator{

protected:
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    
public: 
    //constructor
    ReluLayer(const std::deque<uint32_t> input_shape) {
    
        m_input_shape = input_shape;

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;
    }

    //copy constructor

    //destructor
    virtual ~ReluLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, T *output_tensor) {
        
        /*
        ONNX notation:
        X = input
        Y = output
        */

        utilityTensor::copy_data(m_input_size, input_tensor, output_tensor);

        for(size_t i = 0; i < m_input_size; i++) {
            if(output_tensor[i] < 0) {
                output_tensor[i] = 0;
            }
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_output_tensor,
        T *input_tensor_grad) {

        utilityTensor::copy_data(m_input_size, nextop_grad, input_tensor_grad);

        for(size_t i = 0; i < m_input_size; i++) {
            if(fwd_input_tensor[i] < 0) {
                input_tensor_grad[i] = 0;
            }
        }

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

    return new ReluLayer<DTYPE>(input_tensor_shape);
}

D500_REGISTER_OP(ReluLayer<DTYPE>);