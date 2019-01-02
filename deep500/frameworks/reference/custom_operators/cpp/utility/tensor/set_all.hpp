//set_all.hpp
#ifndef SET_ALL_HPP
#define SET_ALL_HPP

#include <cstring>
#include <algorithm>

//set all elements of Tensor to 0

namespace utilityTensor {
    template <class T>
    void set_all_zero(const int size, T *a) {
        
        std::memset(a, 0, size * sizeof(T));

        return;
    }

    template <class T>
    void set_all(const int size, T *a, T value) {
        
        std::fill(a, a + size, value);

        return;
    }
}

#endif