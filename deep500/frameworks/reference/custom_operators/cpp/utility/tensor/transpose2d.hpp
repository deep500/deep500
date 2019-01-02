//transpose2d.hpp
#ifndef TRANSPOSE2D_HPP
#define TRANSPOSE2D_HPP

#include "copy_data.hpp"

//matrix transposition

namespace utilityTensor {
    template <class T>
    void transpose2d(const int size_m, const int size_n, T *a) {
      
        T a_copy[size_m * size_n];
        utilityTensor::copy_data(size_m * size_n, a, a_copy);

        //#pragma omp parallel for
        for(size_t l = 0; l < size_m * size_n; l++) {
            size_t i = l / size_n;
            size_t j = l % size_n;

            //note: l = i * size_n + j
            a[j * size_m + i] = a_copy[l];
        }

        return;
    }
}

#endif