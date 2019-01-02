//broadcast.hpp
#ifndef BROADCAST_HPP
#define BROADCAST_HPP

#include <stdexcept>
#include "deep500/deep500.h"
#include "copy_data.hpp"
#include "set_all.hpp"
#include "replicate.hpp"

namespace utilityTensor {
    template <class T>
    void unidir_broadcast2d(const uint32_t size_Ma, const uint32_t size_Na,
        const uint32_t size_Mb, const uint32_t size_Nb,
        const T *b, T *b_broadcasted) {

        //returns tensor b unidirectional broadcasted to tensor a

        //tensor b has already the desired shape
        if(size_Ma == size_Mb && size_Na == size_Nb) {
            utilityTensor::copy_data(size_Ma * size_Na, b, b_broadcasted);
            return;
        }

#ifdef DEBUG
        if((size_Mb == size_Ma || size_Mb == 1) && (size_Nb == size_Na) || size_Nb == 1) {
            throw std::runtime_error("tensor b cannot be broadcasted to tensor a")
        }
#endif

        //simple case: tensor b has exactly 1 element: do it efficiently
        if(size_Mb == 1 && size_Nb == 1) {
            utilityTensor::set_all(size_Ma * size_Na, b_broadcasted, b[0]);
            return;
        }

        //tensor B needs broadcasting along 1 dimension only
        if(size_Mb == 1) {
            utilityTensor::replicate2d(size_Ma, 0, size_Nb, b, b_broadcasted);
            return;

        } else {
            utilityTensor::replicate2d(size_Na, 1, size_Mb, b, b_broadcasted);
            return;
        }
    }
}

#endif