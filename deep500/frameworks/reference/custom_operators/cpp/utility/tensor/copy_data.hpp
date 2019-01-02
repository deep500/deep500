//copy_data.hpp
#ifndef COPY_DATA_HPP
#define COPY_DATA_HPP

#include <cstring>

//copy data of tensor efficiently

namespace utilityTensor {
    template <class T>
    void copy_data(const int size, const T *source, T *dest) {
        
        std::memcpy(dest, source, size * sizeof(T));

        return;
    }
}

#endif