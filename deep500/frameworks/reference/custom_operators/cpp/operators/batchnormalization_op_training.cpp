#include <deque>
#include "deep500/deep500.h"
#include "../utility/tensor/copy_data.hpp"
#include "../utility/tensor/set_all.hpp"
#include "../utility/math/mean.hpp"
#include "../utility/math/variance.hpp"
#include <iostream>

//batchnormalization - training mode

template<typename T>
class BatchNormalizationLayerTraining : public deep500::CustomOperator{

protected:
    const float m_epsilon;
    const float m_momentum;
    const int m_spatial;
    int m_initalized; //running mean and running var are init. in first call of forward
    const std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    
public: 
    //constructor
    BatchNormalizationLayerTraining(
        const std::deque<uint32_t> input_shape,
        const float epsilon,
        const float momentum,
        const int spatial) :
        m_epsilon(epsilon),
        m_momentum(momentum),
        m_spatial(spatial),
        m_input_shape(input_shape) {

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;
    }

    //copy constructor

    //destructor
    virtual ~BatchNormalizationLayerTraining() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(
        const T *input_tensor,
        const T *input_tensor2,
        const T *input_tensor3,
        const T *input_tensor4,
        const T *input_tensor5,
        T *output_tensor,
        T *output_tensor2,
        T *output_tensor3,
        T *output_tensor4,
        T *output_tensor5) {
        
        /*
        ONNX notation:
        X = input1
        scale = input2
        B = input3
        mean = input4
        var = input5
        Y = output
        */

        //compute mean of minibatch
        utilityMath::compute_mean(m_input_shape, input_tensor, output_tensor4);

        //compute variance of minibatch
        utilityMath::compute_variance(m_input_shape, input_tensor, output_tensor4, output_tensor5);

        for(size_t i = 0; i < m_input_shape[1]; i++) {
            //compute running mean
            output_tensor2[i] = input_tensor2[i] * m_momentum + input_tensor4[i] * (1 - m_momentum);

            //compute running var
            output_tensor3[i] = input_tensor3[i] * m_momentum + input_tensor5[i] * (1 - m_momentum);
        }

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
        //m_input_shape[1] = size of saved_mean
        T sqrtvar[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            sqrtvar[i] = std::sqrt(output_tensor5[i] + m_epsilon);
        }

        //compute output
        //assume: T output_tensor[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    output_tensor[initial_offset + k * spacing + l] = 
                    input_tensor2[i] * (input_tensor[initial_offset + k * spacing + l] -
                    output_tensor4[i]) / sqrtvar[i] + input_tensor3[i];
                }
            }
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *nextop_grad2,
        const T *nextop_grad3,
        const T *nextop_grad4,
        const T *nextop_grad5,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_input_tensor3,
        const T *fwd_input_tensor4,
        const T *fwd_input_tensor5,
        const T *fwd_output_tensor,
        const T *fwd_output_tensor2,
        const T *fwd_output_tensor3,
        const T *fwd_output_tensor4,
        const T *fwd_output_tensor5,
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

        //number of elements in minibatch per channel
        num_elements = num_blocks * length_block;
        spacing = m_input_shape[1] * length_block;

        T x_min_mu[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    x_min_mu[initial_offset + k * spacing + l] = 
                    fwd_input_tensor[initial_offset + k * spacing + l] - fwd_output_tensor4[i];
                }
            }
        }

        T sqrtvar[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            sqrtvar[i] =  std::sqrt(fwd_output_tensor5[i] + m_epsilon);
        }
        
        T ivar[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            ivar[i] =  1. / sqrtvar[i];        
        }

        T x_hat[m_input_size];
        T d_x_hat[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    x_hat[initial_offset + k * spacing + l] = x_min_mu[initial_offset + k * spacing + l] * ivar[i];
                    d_x_hat[initial_offset + k * spacing + l] = fwd_input_tensor2[i] * nextop_grad[initial_offset + k * spacing + l];
                }
            }
        }
        
        T d_w[m_input_shape[1]];
        T d_b[m_input_shape[1]];
        utilityTensor::set_all_zero(m_input_shape[1], d_w);
        utilityTensor::set_all_zero(m_input_shape[1], d_b);
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_w[i] += nextop_grad[initial_offset + k * spacing + l] * x_hat[initial_offset + k * spacing + l];
                    d_b[i] += nextop_grad[initial_offset + k * spacing + l]; 
                }
            }
            
            
        }

        T d_ivar[m_input_shape[1]];
        utilityTensor::set_all_zero(m_input_shape[1], d_ivar);
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_ivar[i] += d_x_hat[initial_offset + k * spacing + l] * x_min_mu[initial_offset + k * spacing + l];
                }
            }
        }

        T d_xmu1[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_xmu1[initial_offset + k * spacing + l] = d_x_hat[initial_offset + k * spacing + l] * ivar[i];
                }
            }
        }

        T d_sqrtvar[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            d_sqrtvar[i] = -1. / std::pow(sqrtvar[i], 2) * d_ivar[i];
            
            
        }
        
        T d_var[m_input_shape[1]];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            d_var[i] = 0.5 * 1. / sqrtvar[i] * d_sqrtvar[i];
            
            
        }

        T d_sq[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_sq[initial_offset + k * spacing + l] = 1. / (T) num_elements * d_var[i];
                }
            }
        }

        T d_xmu2[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_xmu2[initial_offset + k * spacing + l] = 2 * x_min_mu[initial_offset + k * spacing + l] * d_sq[initial_offset + k * spacing + l];
                }
            }
        }

        T d_x1[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_x1[initial_offset + k * spacing + l] = d_xmu1[initial_offset + k * spacing + l] + d_xmu2[initial_offset + k * spacing + l];
                }
            }
        }

        T d_mu[m_input_shape[1]];
        utilityTensor::set_all_zero(m_input_shape[1], d_mu);
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_mu[i] += (-1.) * (d_xmu1[initial_offset + k * spacing + l] + d_xmu2[initial_offset + k * spacing + l]);
                }
            }
        }

        T d_x2[m_input_size];
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    d_x2[initial_offset + k * spacing + l] = 1. / (T) num_elements * d_mu[i];
                }
            }
        }

        //bwd_params
        for(size_t i = 0; i < m_input_shape[1]; i++) {
            initial_offset = i * length_block;
            
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    input_tensor2_grad[i] += nextop_grad[initial_offset + k * spacing + l] * (fwd_input_tensor[
                        initial_offset + k * spacing + l] - fwd_output_tensor4[i]) / sqrtvar[i];
                    input_tensor3_grad[i] += nextop_grad[initial_offset + k * spacing + l];
                }
            }
        }

        for(size_t i = 0; i < m_input_size; i++) {
            input_tensor_grad[i] = d_x1[i] + d_x2[i];
        }

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

    return new BatchNormalizationLayerTraining<DTYPE>(input_tensor_shape, EPSILON, MOMENTUM, SPATIAL);
}

D500_REGISTER_OP(BatchNormalizationLayerTraining<DTYPE>);