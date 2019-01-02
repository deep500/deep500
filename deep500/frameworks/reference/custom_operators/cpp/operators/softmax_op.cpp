#include <cmath>
#include <deque>
#include <limits>
#include "deep500/deep500.h"
#include "../utility/tensor/set_all.hpp"

//softmaxlayer

template<typename T>
class SoftmaxLayer : public deep500::CustomOperator{

protected:
    const int m_axis;
    const std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    uint32_t m_N;
    uint32_t m_D;
    
public: 
    //constructor
    SoftmaxLayer(
        const std::deque<uint32_t> input_shape,
        const int axis) :
        m_axis(axis),
        m_input_shape(input_shape) {

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;

        //compute N = batch size
        m_N = 1;
        for(size_t i = 0; i < axis; i++) {
            m_N *= input_shape[i];
        }

        //compute D = input feature dimensions
        m_D = 1;
        for(size_t i = axis; i < input_shape.size(); i++) {
            m_D *= input_shape[i];
        }
    }

    //copy constructor

    //destructor
    virtual ~SoftmaxLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(
        const T *input_tensor,
        T *output_tensor) {
        
        /*
        ONNX notation:
        input = input1
        output = output
        */

        T max_x[m_N];
        utilityTensor::set_all(m_N, max_x, std::numeric_limits<T>::lowest());
        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                if(max_x[i] < input_tensor[i * m_D + j]) {
                    max_x[i] = input_tensor[i * m_D + j];
                }
            }
        }

        T exp_x[m_input_size];
        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                exp_x[i * m_D + j] = std::exp(input_tensor[i * m_D + j] - max_x[i]);
            }
        }

        T sum_exp_x[m_N];
        utilityTensor::set_all_zero(m_N, sum_exp_x);
        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                sum_exp_x[i] += exp_x[i * m_D + j];
            }
        }

        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                output_tensor[i * m_D + j] = exp_x[i * m_D + j] / sum_exp_x[i];
            }
        }        

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_output_tensor,
        T *input_tensor_grad) {

        T sum_input_tensor_grad[m_N];
        utilityTensor::set_all_zero(m_N, sum_input_tensor_grad);
        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                sum_input_tensor_grad[i] += nextop_grad[i * m_D + j] * fwd_output_tensor[i * m_D + j];
            }
        }

        for(size_t i = 0; i < m_N; i++) {
            for(size_t j = 0; j < m_D; j++) {
                input_tensor_grad[i * m_D + j] = fwd_output_tensor[i * m_D + j] * 
                    (nextop_grad[i * m_D + j] - sum_input_tensor_grad[i]);
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

    return new SoftmaxLayer<DTYPE>(input_tensor_shape, AXIS);
}

D500_REGISTER_OP(SoftmaxLayer<DTYPE>);