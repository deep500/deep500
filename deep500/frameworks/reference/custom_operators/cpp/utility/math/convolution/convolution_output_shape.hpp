//convolution_output_shape.hpp
#ifndef CONVOLUTION_OUTPUT_SHAPE_HPP
#define CONVOLUTION_OUTPUT_SHAPE_HPP

#include <cmath>
#include <deque>
#include "autopadding.hpp"

//compute the output tensor shape of a convolution

namespace utilityMath {
    void compute_conv_output_shape(
        const std::deque<uint32_t>& input_shape,
        const utilityMath::auto_pad_code auto_pad,
        const uint32_t num_featuremaps,
        const std::deque<uint32_t>& kernel_shape,
        const std::deque<uint32_t>& dilations,
        const std::deque<uint32_t>& pads,
        const std::deque<uint32_t>& strides,
        std::deque<uint32_t>& output_shape) {
        
        const uint8_t kernel_rank = kernel_shape.size();

        const uint32_t N = input_shape[0];
        std::deque<uint32_t> input_spatial_shape = std::deque<uint32_t>(input_shape);
        input_spatial_shape.pop_front();
        input_spatial_shape.pop_front();

        const uint32_t N_out = N;
        const uint32_t C_out = num_featuremaps;
        output_shape.clear();
        output_shape.push_back(N_out);
        output_shape.push_back(C_out);

        uint32_t temp;
        switch(auto_pad) {
            case enum_same_u:
            case enum_same_l:
                for(size_t i = 0; i < kernel_rank; i++) {
                    temp = std::ceil((float)input_spatial_shape[i] / (float)strides[i]) 
                    - (kernel_shape[i] - 1) * (dilations[i] - 1);
                    output_shape.push_back(temp);
                }
                break;
            
            case enum_valid:
                for(size_t i = 0; i < kernel_rank; i++) {
                    temp = std::ceil(
                        (float) input_spatial_shape[i] - \
                        (kernel_shape[i] + (kernel_shape[i] - 1) * (dilations[i] - 1)) / \
                        (float)strides[i]
                    );
                    output_shape.push_back(temp);
                }
                break;

            default:
                for(size_t i = 0; i < kernel_rank; i++) {
                    temp = std::floor(
                        (float) (
                            input_spatial_shape[i] - \
                            kernel_shape[i] - \
                            (kernel_shape[i] - 1) * (dilations[i] - 1) + \
                            pads[i] + \
                            pads[i + kernel_rank]
                        ) / \
                        (float) (
                            strides[i]
                        ) + 1
                    );
                    output_shape.push_back(temp);
                }
                break;

        }

        return;
    }
}

#endif