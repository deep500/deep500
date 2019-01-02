//variance.hpp
#ifndef VARIANCE_HPP
#define VARIANCE_HPP

#include <cstdint>
#include <cmath>
#include <deque>
#include "../tensor/set_all.hpp"

//compute for each channel of the minibatch the variance

namespace utilityMath {
    template <class T>
    void compute_variance(
        const std::deque<uint32_t> input_shape,
        const T *input,
        const T *mean,
        T *output) {

        //compute numpber of elements in input
        size_t input_size = 1;
        for(const auto& elem: input_shape)
            input_size *= elem;

        utilityTensor::set_all_zero(input_shape[1], output);

        size_t num_blocks;
        size_t length_block;
        size_t num_elements;
        size_t spacing;
        size_t initial_offset;
        size_t temp1;
        T temp2;

        num_blocks = input_shape[0];

        temp1 = 1;
        for(size_t j = 2; j < input_shape.size(); j++) {
            temp1 *= input_shape[j];
        }
        length_block = temp1;
        num_elements = num_blocks * length_block;
        spacing = input_shape[1] * length_block;

        for(size_t i = 0; i < input_shape[1]; i++) {
            initial_offset = i * length_block;
            
            temp2 = 0;
            for(size_t k = 0; k < num_blocks; k++) {
                for(size_t l = 0; l < length_block; l++) {
                    temp2 += std::pow(input[initial_offset + k * spacing + l] - mean[i], 2);
                }
            }
            temp2 /= (T) (num_elements);
            output[i] = temp2;
        }

        return;  
    }
}

#endif