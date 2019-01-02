#include <deque>
#include "deep500/deep500.h"
#include "../utility/tensor/copy_data.hpp"
#include "../utility/tensor/set_all.hpp"

//batchnormalization - inference mode

template<typename T>
class BatchNormalizationLayerInference : public deep500::CustomOperator{

protected:
    float m_epsilon;
    float m_momentum;
    int m_spatial;
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    
public: 
    //constructor
    BatchNormalizationLayerInference(
        const std::deque<uint32_t> input_shape,
        const float epsilon,
        const float momentum,
        const int spatial) {
    
        m_epsilon = epsilon;
        m_momentum = m_momentum;
        m_spatial = spatial;
        m_input_shape = input_shape;

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;
    }

    //copy constructor

    //destructor
    virtual ~BatchNormalizationLayerInference() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(
        const T *input_tensor,
        const T *input_tensor2,
        const T *input_tensor3,
        const T *input_tensor4,
        const T *input_tensor5,
        T *output_tensor) {
        
        /*
        ONNX notation:
        X = input1
        scale = input2
        B = input3
        mean = input4
        var = input5
        Y = output
        */

        size_t num_blocks;
        size_t length_block;
        size_t num_elements;
        size_t spacing;
        size_t initial_offset;
        size_t temp;

        num_blocks = m_input_shape[0];

        temp = 1;
        for(size_t j = 2; j < m_input_shape.size(); j++) {
            temp *= m_input_shape[j];
        }
        length_block = temp;

        num_elements = num_blocks * length_block;
        spacing = m_input_shape[1] * length_block;

        //compute standard deviation of minibatch
        //m_input_shape[1] = size of m_saved_mean
        T sqrtvar[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            sqrtvar[i] = std::sqrt(input_tensor5[i] + m_epsilon);
        }

        //compute output
        //assume: T output_tensor[m_input_size]; 
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    output_tensor[initial_offset + k * spacing + l] = 
                    input_tensor2[i] * (input_tensor[initial_offset + k * spacing + l] -
                    input_tensor4[i]) / sqrtvar[i] + input_tensor3[i];
                }
            }
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_input_tensor3,
        const T *fwd_input_tensor4,
        const T *fwd_input_tensor5,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad,
        T *input_tensor3_grad,
        T *input_tensor4_grad,
        T *input_tensor5_grad) {

        utilityTensor::set_all_zero(m_input_size, input_tensor_grad);
        utilityTensor::set_all_zero(m_input_shape[1], input_tensor2_grad);
        utilityTensor::set_all_zero(m_input_shape[1], input_tensor3_grad);
        utilityTensor::set_all_zero(m_input_shape[1], input_tensor4_grad);
        utilityTensor::set_all_zero(m_input_shape[1], input_tensor5_grad);

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

    return new BatchNormalizationLayerInference<DTYPE>(input_tensor_shape, EPSILON, MOMENTUM, SPATIAL);
}

D500_REGISTER_OP(BatchNormalizationLayerInference<DTYPE>);