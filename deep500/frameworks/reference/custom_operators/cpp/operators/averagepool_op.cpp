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
class AveragePoolLayer : public deep500::CustomOperator{

protected:
    utilityMath::auto_pad_code m_auto_pad;
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    std::deque<uint32_t> m_kernel_shape;
    uint8_t m_kernel_rank;
    uint32_t m_kernel_size;
    std::deque<uint32_t> m_pads;
    std::deque<uint32_t> m_strides;
    std::deque<uint32_t> m_output_shape;
    uint8_t m_output_rank;
    uint32_t m_output_size;
    int m_count_include_pad;
    
public: 
    //constructor
    AveragePoolLayer(const std::deque<uint32_t> input_shape, 
        const std::string auto_pad,
        const std::deque<uint32_t> kernel_shape,
        const std::deque<uint32_t> pads,
        const std::deque<uint32_t> strides,
        const int count_include_pad) :
        m_auto_pad(utilityMath::string_to_auto_pad_code(auto_pad)),
        m_input_shape(input_shape),
        m_kernel_shape(kernel_shape),
        m_kernel_rank(m_kernel_shape.size()),
        m_count_include_pad(count_include_pad) {

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;

        //compute numpber of elements in kernel
        m_kernel_size = 1;
        for(const auto& elem: kernel_shape)
            m_kernel_size *= elem;

        //if no strides are specified, initalize with default values
        if(strides.size() > 0) { 
            m_strides = strides;
        } else {
            m_strides.clear();
            for(size_t i = 0; i < m_kernel_rank; i++) {
                m_strides.push_back(1);
            }
        }

        //if no spads are specified, initalize with default values
        if(pads.size() > 0) { 
            m_pads = pads;
        } else {
            m_pads.clear();
            for(size_t i = 0; i < m_kernel_rank; i++) {
                m_pads.push_back(0);
                m_pads.push_back(0);
            }
        }

        //compute the shape of the output
        std::deque<uint32_t> dilations_temp;
        for(size_t i = 0; i < m_kernel_rank; i++) {
            dilations_temp.push_back(1);
        }
        utilityMath::compute_conv_output_shape(
            m_input_shape,
            m_auto_pad,
            m_input_shape[1],
            m_kernel_shape,
            dilations_temp,
            pads,
            m_strides,
            m_output_shape
        );

        //compute padding if necessary
        switch(m_auto_pad) {
            case utilityMath::enum_explicit:
                m_pads = pads;
                break;
                
            case utilityMath::enum_same_u:
            case utilityMath::enum_same_l:
                utilityMath::auto_pad(
                    m_input_shape,
                    m_auto_pad,
                    m_kernel_shape,
                    dilations_temp,
                    m_strides,
                    m_output_shape,
                    m_pads
                );
                break;

            case utilityMath::enum_valid:
                for(size_t i = 0; i < m_kernel_rank; i++) {
                    m_pads.clear();
                    m_pads.push_back(0);
                    m_pads.push_back(0);
                }
                break;
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
        std::deque<uint32_t>::iterator it = m_pads.begin() + m_kernel_rank;
        for(size_t i = 0; i < 5 - m_output_rank; i++) {
            m_input_shape.push_back(1);
            m_kernel_shape.push_back(1);
            m_output_shape.push_back(1);
            //m_pads: insert in the middle & at the end
            m_pads.insert(it, 0);
            it++;
            m_pads.push_back(0);
            m_strides.push_back(1);
        }
    }

    //copy constructor

    //destructor
    virtual ~AveragePoolLayer() {}

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
        size_t temp_n;
        if(m_count_include_pad) {
            temp_n = 1;
            for(size_t i = 0; i < m_kernel_rank; i++) {
                temp_n *= m_kernel_shape[i];
            }
        }

        //loop over output tensor
        for(size_t i = 0; i < m_output_shape[0]; i++) {
            for(size_t j = 0; j < m_output_shape[1]; j++) {
                for(size_t k = 0; k < m_output_shape[2]; k++) {
                    for(size_t l = 0; l < m_output_shape[3]; l++) {
                        for(size_t m = 0; m < m_output_shape[4]; m++) {
                            if (!m_count_include_pad) {
                                temp_n = 0;
                                //loop over kernel
                                for(size_t u = 0; u < m_kernel_shape[0]; u++) {
                                    k_input_temp = -m_pads[0] + k * m_strides[0] + u;
                                    if(k_input_temp >= 0 && k_input_temp < m_input_shape[2]) {
                                        for(size_t v = 0; v < m_kernel_shape[1]; v++) {
                                            l_input_temp = -m_pads[1] + l * m_strides[1] + v;
                                            if(l_input_temp >= 0 && l_input_temp < m_input_shape[3]) {
                                                for(size_t w = 0; w < m_kernel_shape[2]; w++) {
                                                    m_input_temp = -m_pads[2] + m * m_strides[2] + w;
                                                    if(m_input_temp >=0 && m_input_shape[4]) {
                                                        temp_n++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } 
                            temp = 0;
                            //loop over kernel
                            for(size_t u = 0; u < m_kernel_shape[0]; u++) {
                                k_input_temp = -m_pads[0] + k * m_strides[0] + u;
                                if(k_input_temp >= 0 && k_input_temp < m_input_shape[2]) {
                                    for(size_t v = 0; v < m_kernel_shape[1]; v++) {
                                        l_input_temp = -m_pads[1] + l * m_strides[1] + v;
                                        if(l_input_temp >= 0 && l_input_temp < m_input_shape[3]) {
                                            for(size_t w = 0; w < m_kernel_shape[2]; w++) {
                                                m_input_temp = -m_pads[2] + m * m_strides[2] + w;
                                                if(m_input_temp >=0 && m_input_shape[4]) {
                                                    temp_index = utilityTensor::compute_index(
                                                        m_input_shape, {i, j,
                                                        (size_t)k_input_temp,
                                                        (size_t)l_input_temp,
                                                        (size_t)m_input_temp}
                                                    );
                                                    temp += input_tensor[temp_index];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            temp_index = utilityTensor::compute_index(
                                m_output_shape, {i, j, k, l, m}
                            );
                            output_tensor[temp_index] = temp / temp_n;
                        }
                    }
                }
            }
        }

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_output_tensor,
        T *input_tensor_grad) {

        utilityTensor::set_all(m_input_size, input_tensor_grad, (T)0.);

        size_t temp_index;
        int k_input_temp;
        int l_input_temp;
        int m_input_temp;
        T temp;
        size_t temp_n;
        if(m_count_include_pad) {
            temp_n = 1;
            for(size_t i = 0; i < m_kernel_rank; i++) {
                temp_n *= m_kernel_shape[i];
            }
        }

        //loop over output tensor
        for(size_t i = 0; i < m_output_shape[0]; i++) {
            for(size_t j = 0; j < m_output_shape[1]; j++) {
                for(size_t k = 0; k < m_output_shape[2]; k++) {
                    for(size_t l = 0; l < m_output_shape[3]; l++) {
                        for(size_t m = 0; m < m_output_shape[4]; m++) {
                            if (!m_count_include_pad) {
                                temp_n = 0;
                                //loop over kernel
                                for(size_t u = 0; u < m_kernel_shape[0]; u++) {
                                    k_input_temp = -m_pads[0] + k * m_strides[0] + u;
                                    if(k_input_temp >= 0 && k_input_temp < m_input_shape[2]) {
                                        for(size_t v = 0; v < m_kernel_shape[1]; v++) {
                                            l_input_temp = -m_pads[1] + l * m_strides[1] + v;
                                            if(l_input_temp >= 0 && l_input_temp < m_input_shape[3]) {
                                                for(size_t w = 0; w < m_kernel_shape[2]; w++) {
                                                    m_input_temp = -m_pads[2] + m * m_strides[2] + w;
                                                    if(m_input_temp >=0 && m_input_shape[4]) {
                                                        temp_n++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            temp_index = utilityTensor::compute_index(
                                m_output_shape, {i, j, k, l, m}
                            );
                            temp = 1. / temp_n * nextop_grad[temp_index];
                            //loop over kernel
                            for(size_t u = 0; u < m_kernel_shape[0]; u++) {
                                k_input_temp = -m_pads[0] + k * m_strides[0] + u;
                                if(k_input_temp >= 0 && k_input_temp < m_input_shape[2]) {
                                    for(size_t v = 0; v < m_kernel_shape[1]; v++) {
                                        l_input_temp = -m_pads[1] + l * m_strides[1] + v;
                                        if(l_input_temp >= 0 && l_input_temp < m_input_shape[3]) {
                                            for(size_t w = 0; w < m_kernel_shape[2]; w++) {
                                                m_input_temp = -m_pads[2] + m * m_strides[2] + w;
                                                if(m_input_temp >=0 && m_input_temp < m_input_shape[4]) {
                                                    temp_index = utilityTensor::compute_index(
                                                        m_input_shape, {i, j,
                                                        (size_t)k_input_temp,
                                                        (size_t)l_input_temp,
                                                        (size_t)m_input_temp}
                                                    );
                                                    input_tensor_grad[temp_index] += temp;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
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
    std::string auto_pad = STRINGIFY(AUTO_PAD);
    std::deque<uint32_t> kernel_shape = {KERNEL_SHAPE};
    std::deque<uint32_t> pads = {PADS};
    std::deque<uint32_t> strides = {STRIDES};

    return new AveragePoolLayer<DTYPE>(input_tensor_shape, auto_pad, kernel_shape, pads, strides, COUNT_INCLUDE_PAD);
}

D500_REGISTER_OP(AveragePoolLayer<DTYPE>);