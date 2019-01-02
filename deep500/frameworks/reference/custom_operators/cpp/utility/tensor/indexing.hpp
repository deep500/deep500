//indexing.hpp
#ifndef INDEXING_HPP
#define INDEXING_HPP

#include <cstdint>
#include <deque>
#include <initializer_list>
//#include <cstdarg>

//when accessing an element of a tensor, computes
//the corresponding index of the array that stores
//its data

namespace utilityTensor {
    template <typename... Args>
    size_t compute_index(const std::deque<uint32_t> shape,
        std::initializer_list<size_t> indices) {

        std::initializer_list<size_t>::iterator it = indices.begin();
        size_t temp = *it;
        it++;

        size_t temp_index = 1;
        for( ; it != indices.end(); it++) {
            temp = (size_t)shape[temp_index] * temp + *it;
            temp_index++;
        }

        return temp;
    }
}

#endif