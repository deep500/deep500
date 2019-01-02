#include <deque>
#include "deep500/deep500.h"
#include "../utility/tensor/copy_data.hpp"
#include "../utility/tensor/set_all.hpp"

template<typename T>
class ReshapeLayer : public deep500::CustomOperator{

protected:
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    int64_t m_output_rank;
    
public: 
    //constructor
    ReshapeLayer(const std::deque<uint32_t> input_shape,
    const int64_t output_rank) {
    
        m_input_shape = input_shape;

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;

        m_output_rank = output_rank;
    }

    //copy constructor

    //destructor
    virtual ~ReshapeLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor,
    const int64_t *input_tensor2,
    T *output_tensor) {
        
        /*
        ONNX notation:
        data = input
        reshaped = output
        */

        utilityTensor::copy_data(m_input_size, input_tensor, output_tensor);

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const int64_t *fwd_input_tensor2,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        int64_t *input_tensor2_grad) {

        utilityTensor::copy_data(m_input_size, nextop_grad, input_tensor_grad);

        utilityTensor::set_all_zero(m_output_rank, input_tensor2_grad);

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

    int64_t output_tensor_rank = input_descriptors[1].sizes[0];



    return new ReshapeLayer<DTYPE>(input_tensor_shape, output_tensor_rank);
}

D500_REGISTER_OP(ReshapeLayer<DTYPE>);