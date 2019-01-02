//autopadding.hpp
#ifndef AUTOPADDING_HPP
#define AUTOPADDING_HPP

#include <cmath>
#include <string>
#include <deque>
#include <stdexcept>

//autopadding according to ONNX definition

namespace utilityMath {

    enum auto_pad_code {
        enum_same_u,
        enum_same_l,
        enum_valid,
        enum_explicit
    };

    const auto_pad_code string_to_auto_pad_code(std::string const input) {
        if(input == "SAME_UPPER") {
            return enum_same_u;
        }else if(input == "SAME_LOWER") {
            return enum_same_l;
        }else if(input == "VALID") {
            return enum_valid;
        }else {
            return enum_explicit;
        }
    };

    void auto_pad(
        const std::deque<uint32_t>& input_shape,
        const auto_pad_code auto_pad,
        const std::deque<uint32_t>& kernel_shape,
        const std::deque<uint32_t>& dilations,
        const std::deque<uint32_t>& strides,
        const std::deque<uint32_t>& output_shape,
        std::deque<uint32_t>& pads) {

        const uint8_t input_rank = input_shape.size();
        const uint8_t kernel_rank = kernel_shape.size();

        std::deque<uint32_t> input_spatial_shape = std::deque<uint32_t>(input_shape);
        input_spatial_shape.pop_front();
        input_spatial_shape.pop_front();

        std::deque<uint32_t> output_spatial_shape = std::deque<uint32_t>(output_shape);
        output_spatial_shape.pop_front();
        output_spatial_shape.pop_front();
        std::deque<uint32_t> pads_tot;
        
        uint32_t temp;
        for(size_t i = 0; i < kernel_rank; i++) {
            temp = (output_spatial_shape[i] - 1) * strides[i] + \
                (kernel_shape[i]) + \
                (kernel_shape[i] - 1) * (dilations[i] - 1) - \
                input_spatial_shape[i];
            pads_tot.push_back(temp);
        }

        switch(auto_pad) {
            case enum_same_u:
                //add extra padding at the end
                pads.clear();
                for(size_t i = 0; i < kernel_rank; i++) {
                    pads.push_back(pads_tot[i] / 2);
                }
                for(size_t i = 0; i < kernel_rank; i++) {
                    pads.push_back((pads_tot[i] + 1) / 2);
                }
                break;

            case enum_same_l:
                //add extra padding at the begining
                pads.clear();
                for(size_t i = 0; i < kernel_rank; i++) {
                    pads.push_back((pads_tot[i] + 1) / 2);
                }
                for(size_t i = 0; i < kernel_rank; i++) {
                    pads.push_back(pads_tot[i] / 2);
                }
                break;

            default:
                throw std::runtime_error("error in fullconv_pad(...)");
                break;

        }

        return;
    }

    void fullconv_pad(
        const std::deque<uint32_t>& weight_shape,
        const std::deque<uint32_t>& dilations,
        std::deque<uint32_t>& pads_fullconv) {

        pads_fullconv.clear();

        uint32_t temp;

        for(int i = 2; i < weight_shape.size(); i++) {
            //compute dilated filter size - 1
            temp = weight_shape[i] + \
                (weight_shape[i] - 1) * \
                (dilations[i - 2] - 1) - \
                1;

            pads_fullconv.push_back(temp);
        }

        //fullconv padding at the end is the same as in the beginnning:
        for(int i = 2; i < weight_shape.size(); i++) {
            temp = weight_shape[i] + \
                (weight_shape[i] - 1) * \
                (dilations[i - 2] - 1) - \
                1;

            pads_fullconv.push_back(temp);
        }
    
        return;
    }
}

#endif