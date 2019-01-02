#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

#include <deque>
#include <string>
#include "deep500/deep500.h"
#include "../utility/math/convolution/autopadding.hpp"
#include "../utility/math/convolution/convolution_output_shape.hpp"
#include "../utility/tensor/set_all.hpp"
#include "../utility/tensor/indexing.hpp"

//currently supports only 1-d, 2-d, and 3-d AveragePooling (i.e. 3-d, 4-d, 5-d tensors)

template<typename T>
class GlobalAveragePoolLayer : public deep500::CustomOperator{

protected:
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    std::deque<uint32_t> m_output_shape;
    uint8_t m_output_rank;
    uint32_t m_output_size;
    uint32_t m_kernel_size;
    
public: 
    //constructor
    GlobalAveragePoolLayer(const std::deque<uint32_t> input_shape) {
    
        m_input_shape = input_shape;

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;

        //compute output shape

        //compute the shape of the output
        m_output_shape.push_back(input_shape[0]);
        m_output_shape.push_back(input_shape[1]);
        for(size_t i = 0; i < m_input_shape.size() - 2; i++) {
            m_output_shape.push_back(1);
        }

        m_output_rank = m_output_shape.size();

        //compute the number of elements in the output tensor
        m_output_size = 1;
        for(const auto& elem: m_output_shape)
            m_output_size *= elem;

        /*
        currently supports only 1d, 2d, and 3d AveragePooling

        AveragePool_1d: 3d-tensor
        AveragePool_2d: 4d-tensor
        AveragePool_3d: 5d-tensor
        */

        //extend with default values to prevent out of bounds access
        for(size_t i = 0; i < 5 - m_output_rank; i++) {
            m_input_shape.push_back(1);
        }

        m_kernel_size = 1;
        for(size_t i = 2; i < m_input_shape.size(); i++) {
            m_kernel_size *= m_input_shape[i];
        }

    }

    //copy constructor

    //destructor
    virtual ~GlobalAveragePoolLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, T *output_tensor) {
        
        /*
        ONNX notation:
        x = input
        y = output
        */

        utilityTensor::set_all(m_output_size, output_tensor, std::numeric_limits<T>::lowest());

        size_t temp_index;
        int k_input_temp;
        int l_input_temp;
        int m_input_temp;
        T temp;
        size_t k = 0;
        size_t l = 0;
        size_t m = 0;

        //loop over output tensor
        for(size_t i = 0; i < m_output_shape[0]; i++) {
            for(size_t j = 0; j < m_output_shape[1]; j++) {
                temp = 0;
                //loop over kernel
                for(size_t u = 0; u < m_input_shape[2]; u++) {
                    for(size_t v = 0; v < m_input_shape[3]; v++) {
                        for(size_t w = 0; w < m_input_shape[4]; w++) {
                            temp_index = utilityTensor::compute_index(
                                m_input_shape, {
                                i,
                                j,
                                u,
                                v,
                                w}
                            );
                            temp += input_tensor[temp_index];
                        } 
                    }
                }
                
                output_tensor[i * m_output_shape[1] + j] = temp / m_kernel_size;
            }
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_output_tensor,
        T *input_tensor_grad) {

        size_t temp_index;
        T temp;

        //loop over grad_output tensor
        for(size_t i = 0; i < m_output_shape[0]; i++) {
            for(size_t j = 0; j < m_output_shape[1]; j++) {
                temp = nextop_grad[i * m_output_shape[1] + j] / m_kernel_size;
                //loop over grad_input
                for(size_t u = 0; u < m_input_shape[2]; u++) {
                    for(size_t v = 0; v < m_input_shape[3]; v++) {
                        for(size_t w = 0; w < m_input_shape[4]; w++) {
                            temp_index = utilityTensor::compute_index(
                                m_input_shape, {
                                i,
                                j,
                                u,
                                v,
                                w}
                            );
                            input_tensor_grad[temp_index] = temp;
                        } 
                    }
                }
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

    return new GlobalAveragePoolLayer<DTYPE>(input_tensor_shape);
}

D500_REGISTER_OP(GlobalAveragePoolLayer<DTYPE>);