#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)

#include <deque>
#include "deep500/deep500.h"
#include "../utility/math/convolution/autopadding.hpp"
#include "../utility/math/convolution/convolution_output_shape.hpp"
#include "../utility/math/convolution/crosscorrelation.hpp"
#include "../utility/math/convolution/crosscorrelation_dilx_flipw.hpp"
#include "../utility/math/convolution/crosscorrelation_swap_axes.hpp"
#include "../utility/tensor/set_all.hpp"
#include "../utility/tensor/indexing.hpp"

//currently supports only 1-d, 2-d, and 3-d convolution (i.e. 3-d, 4-d, 5-d tensors)

template<typename T>
class ConvLayer : public deep500::CustomOperator{

protected:
    utilityMath::auto_pad_code m_auto_pad;
    std::deque<uint32_t> m_input_shape;
    uint32_t m_input_size;
    std::deque<uint32_t> m_weight_shape;
    uint32_t m_weight_size;
    std::deque<uint32_t> m_kernel_shape;
    uint8_t m_kernel_rank;
    uint32_t m_kernel_size;
    std::deque<uint32_t> m_dilations;
    int m_group;
    std::deque<uint32_t> m_pads;
    std::deque<uint32_t> m_strides;
    std::deque<uint32_t> m_output_shape;
    uint8_t m_output_rank;
    uint32_t m_output_size;
    
public: 
    //constructor
    ConvLayer(const std::deque<uint32_t> input_shape,
        const std::deque<uint32_t> weight_shape,
        const std::string auto_pad,
        const std::deque<uint32_t> dilations,
        const int group,
        const std::deque<uint32_t> kernel_shape,
        const std::deque<uint32_t> pads,
        const std::deque<uint32_t> strides) :
        m_auto_pad(utilityMath::string_to_auto_pad_code(auto_pad)),
        m_group(group),
        m_input_shape(input_shape),
        m_weight_shape(weight_shape){

        //compute numpber of elements in input
        m_input_size = 1;
        for(const auto& elem: input_shape)
            m_input_size *= elem;

        //compute numpber of elements in weight tensor
        m_weight_size = 1;
        for(const auto& elem: weight_shape)
            m_weight_size *= elem;

        //compute numpber of elements in kernel
        m_kernel_size = 1;
        for(const auto& elem: kernel_shape)
            m_kernel_size *= elem;

        //if no kernel shape is specified, infer from weight tensor
        if(kernel_shape.size() > 0) {
            m_kernel_shape = kernel_shape;
        } else {
            m_kernel_shape = weight_shape;
            m_kernel_shape.pop_front();
            m_kernel_shape.pop_front();
        }
        m_kernel_rank = m_kernel_shape.size();

        //if no dilation is specified, initalize with default values
        if(dilations.size() > 0) {
            m_dilations = dilations;
        } else {
            for(size_t i = 0; i < m_kernel_rank; i++) {
                m_dilations.push_back(1);
            }
        }
        //if no strides are specified, initalize with default values
        if(strides.size() > 0) { 
            m_strides = strides;
        } else {
            for(size_t i = 0; i < m_kernel_rank; i++) {
                m_strides.push_back(1);
            }
        }

        //compute the shape of the output
        utilityMath::compute_conv_output_shape(
            m_input_shape,
            m_auto_pad,
            m_input_shape[1],
            m_kernel_shape,
            m_dilations,
            pads,
            m_strides,
            m_output_shape);

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
                    m_dilations,
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

        //compute the shape of the output
        utilityMath::compute_conv_output_shape(
            m_input_shape,
            m_auto_pad,
            //m_input_shape[1],
            m_weight_shape[0],
            m_kernel_shape,
            m_dilations,
            m_pads,
            m_strides,
            m_output_shape);

        m_output_rank = m_output_shape.size();

        //compute the number of elements in the output tensor
        m_output_size = 1;
        for(const auto& elem: m_output_shape)
            m_output_size *= elem;

        /*
        currently supports only 1d, 2d, and 3d convolution

        conv_1d: 3d-tensor
        conv_2d: 4d-tensor
        conv_3d: 5d-tensor
        */

        //extend with default values to prevent out of bounds access in the for loops
        std::deque<uint32_t>::iterator it = m_pads.begin() + m_kernel_rank;
        for(size_t i = 0; i < 5 - m_output_rank; i++) {
            m_input_shape.push_back(1);
            m_weight_shape.push_back(1);
            m_output_shape.push_back(1);
            m_dilations.push_back(1);
            //m_pads: insert in the middle & at the end
            m_pads.insert(it, 0);
            it++;
            m_pads.push_back(0);
            m_strides.push_back(1);
        }
    }

    //copy constructor

    //destructor
    virtual ~ConvLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, const T *input_tensor2, const T *input_tensor3, T *output_tensor) {
        
        /*
        ONNX notation:
        X = input_tensor
        W = input_tensor2
        B = input_tensor3
        Y = output_tensor
        */
        
        utilityMath::crosscorrelation_bias(
            m_input_shape,
            m_output_shape,
            m_group,
            m_weight_shape,
            m_dilations,
            m_pads,
            m_strides,
            input_tensor,
            input_tensor2,
            input_tensor3,
            output_tensor);

        return;
    }

    void forward(const T *input_tensor, const T *input_tensor2, T *output_tensor) {  
             
        /*
        ONNX notation:
        X = input_tensor
        W = input_tensor2
        Y = output_tensor
        */
       //printf("In forward method with 2 inputs\n");

        utilityMath::crosscorrelation(
            m_input_shape,
            m_output_shape,
            m_group,
            m_weight_shape,
            m_dilations,
            m_pads,
            m_strides,
            input_tensor,
            input_tensor2,
            output_tensor);

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_input_tensor3,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad,
        T *input_tensor3_grad) {

        /*
        //m = number of feature maps of weight
        //c = number of channels / group of weight
        //kh = height of kernel
        //kw = width of kernel

        //N = batch size of input
        //C = number of channels of input
        //H = height of input
        //W = width of input

        //N_out = batch size of output
        //C_out = number of channels of output
        //H_out = height of output
        //W_out = width of output
        */

        //set input gradient tensor to zero
        utilityTensor::set_all_zero(m_input_size, input_tensor_grad);
        //set weight (= kernel) gradient tensor to zero
        utilityTensor::set_all_zero(m_weight_size, input_tensor2_grad);
        //set bias gradient tensor to zero
        utilityTensor::set_all_zero(m_weight_shape[0], input_tensor3_grad);

        //kernel shape of bprop_nextlayer
        std::deque<uint32_t> kernel_shape_temp = std::deque<uint32_t>(m_output_shape);
        kernel_shape_temp.pop_front();
        kernel_shape_temp.pop_front();

       //prepare full convolution in order to compute gradient of input
       std::deque<uint32_t> pads_fullconv;
       utilityMath::fullconv_pad(
           m_weight_shape,
           m_dilations,
           pads_fullconv);
        for(size_t i = 0; i < m_weight_shape.size() - 2; i++) {
            pads_fullconv[i] -= m_pads[i];
            pads_fullconv[i + 3] -= m_pads[i + 3];
        }

        //compute input gradient
        //note: pad for 'full convolution'
        //note: swapped dilations and strides
        //note: no bias
        //note: kernel is flipped by 180 degrees by using convolution istead of crosscorrelation
        //note: crosscorrelation_dilx_flipw swapes axes 0 and 1 of W during computation
        std::deque<uint32_t> strides_temp = {1, 1, 1};
        
        utilityMath::crosscorrelation_dilx_flipw(
            m_output_shape,
            m_input_shape,
            m_group,
            m_weight_shape,
            m_dilations,
            pads_fullconv,
            strides_temp,
            m_strides,
            nextop_grad,
            fwd_input_tensor2,
            input_tensor_grad);

        //prepare convolution in order to compute weight gradient
        //swap dilations and strides:
        //swap dilations and strides(before computing padding!)
        std::deque<uint32_t> dilations_temp = std::deque<uint32_t>(m_strides);
        strides_temp = std::deque<uint32_t>(m_dilations);
        //compute weight gradient, don't use bias
        //utilityMath::crosscorrelation(
        utilityMath::crosscorrelation_swap_axes(
            m_input_shape,
            m_weight_shape,
            m_group,
            m_output_shape,
            dilations_temp,
            m_pads,
            strides_temp,
            fwd_input_tensor,
            nextop_grad,
            input_tensor2_grad);
       
        size_t index_temp;
        //loop over output gradient tensor
        for(size_t i = 0; i < m_output_shape[0]; i++) {
            for(size_t j = 0; j < m_output_shape[1]; j++) {
                for(size_t k = 0; k < m_output_shape[2]; k++) {
                    for(size_t l = 0; l < m_output_shape[3]; l++) {
                        for(size_t m = 0; m < m_output_shape[4]; m++) {
                            index_temp = utilityTensor::compute_index(
                                m_output_shape,
                                {i,
                                j,
                                k,
                                l,
                                m});
                            input_tensor3_grad[j] += nextop_grad[index_temp];
                            
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
        const T *fwd_input_tensor2,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad) {

        //set input gradient tensor to zero
        utilityTensor::set_all_zero(m_input_size, input_tensor_grad);
        //set weight (= kernel) gradient tensor to zero
        utilityTensor::set_all_zero(m_weight_size, input_tensor2_grad);

        //kernel shape of bprop_nextlayer
        std::deque<uint32_t> kernel_shape_temp = std::deque<uint32_t>(m_output_shape);
        kernel_shape_temp.pop_front();
        kernel_shape_temp.pop_front();

/*
        //compute pads for full convolution
        std::deque<uint32_t> pads_fullconv_temp = std::deque<uint32_t>(m_pads);
 
        utilityMath::fullconv_pad(
            m_weight_shape,
            m_input_shape,
            m_auto_pad,
            kernel_shape_temp,
            dilations_temp, 
            strides_temp,
            pads_fullconv_temp);
        
        //compute input gradient
        //note: pad for 'full convolution'
        //note: swapped dilations and strides
        //note: no bias
        //note: kernel is flipped by 180 degrees by using convolution istead of crosscorrelation
        utilityMath::crosscorrelation_rotx(
            m_weight_shape,
            m_input_shape,
            m_group,
            m_output_shape,
            dilations_temp,
            pads_fullconv_temp,
            strides_temp,
            fwd_input_tensor2,
            nextop_grad,
            input_tensor_grad);
*/
        std::deque<uint32_t> pads_fullconv;
        utilityMath::fullconv_pad(
           m_weight_shape,
           m_dilations,
           pads_fullconv);
        for(size_t i = 0; i < m_weight_shape.size() - 2; i++) {
            pads_fullconv[i] -= m_pads[i];
            pads_fullconv[i + 3] -= m_pads[i + 3];
        }

            //compute input gradient
            //note: pad for 'full convolution'
            //note: swapped dilations and strides
            //note: no bias
            //note: kernel is flipped by 180 degrees by using convolution istead of crosscorrelation
            std::deque<uint32_t> strides_temp = {1, 1, 1};

            utilityMath::crosscorrelation_dilx_flipw(
                m_output_shape,
                m_input_shape,
                m_group,
                m_weight_shape,
                m_dilations,
                pads_fullconv,
                strides_temp,
                m_strides,
                nextop_grad,
                fwd_input_tensor2,
                input_tensor_grad);

        //prepare convolution in order to compute weight gradient
        //swap dilations and strides:
        //already done
        //swap dilations and strides(before computing padding!)
        std::deque<uint32_t> dilations_temp = std::deque<uint32_t>(m_strides);
        strides_temp = std::deque<uint32_t>(m_dilations);
        //compute weight gradient, don't use bias
        //utilityMath::crosscorrelation(
        utilityMath::crosscorrelation_swap_axes(
            m_input_shape,
            m_weight_shape,
            m_group,
            m_output_shape,
            dilations_temp,
            m_pads,
            strides_temp,
            fwd_input_tensor,
            nextop_grad,
            input_tensor2_grad);

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
    std::deque<uint32_t> input_tensor2_shape(
        input_descriptors[1].sizes,
        input_descriptors[1].sizes + input_descriptors[1].dims
    );

    std::string auto_pad = STRINGIFY(AUTO_PAD);
    std::deque<uint32_t> dilations = {DILATIONS};
    std::deque<uint32_t> kernel_shape = {KERNEL_SHAPE};
    std::deque<uint32_t> pads = {PADS};
    std::deque<uint32_t> strides = {STRIDES};

    return new ConvLayer<DTYPE>(input_tensor_shape, input_tensor2_shape, auto_pad, dilations, GROUP, kernel_shape, pads, strides);
}

D500_REGISTER_OP(ConvLayer<DTYPE>);