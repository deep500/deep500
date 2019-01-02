//replicate.hpp
#ifndef REPLICATE_HPP
#define REPLICATE_HPP

#include <stdexcept>
#include <algorithm>
#include "deep500/deep500.h"

//replicate a tensor along a given dimension

namespace utilityTensor {
    template <class T>
    void replicate2d(const int n, const int dim,
        const uint32_t size,
        const T *source, T *replicated) {

        //assumes source is either a 1d tensor or a 2d tensor with at least one dimension size = 1
        if(dim == 0) {
            for(size_t i = 0; i < size; i++) {
                std::fill(replicated + i * n, replicated + (i + 1) * n, source[i]);
            }
        } else if(dim == 1) {
            for(size_t i = 0; i < n; i++) {
                std::memcpy(replicated + i * size, source, size * sizeof(T));
            }
        } else {
            throw std::runtime_error("replicate2d can only be used along the first or the second dimension");
        }

        return;
    }
}

#endif