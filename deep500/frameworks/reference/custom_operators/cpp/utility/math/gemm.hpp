//gemm.hpp
#ifndef GEMM_HPP
#define GEMM_HPP

//general matrix-matrix-multiplication

namespace utilityMath {
    template <class T>
    void gemm(const int transA, const int transB, const int size_m, const int size_n, const int size_k,
    const float alpha, const T *a, const T *b, const float beta, T *c) {

        T temp;
        if(!transA && !transB) {
            for(size_t i = 0; i < size_m; i++) {
                for(size_t j = 0; j < size_n; j++) {
                    temp = 0;
                    for(size_t k = 0; k < size_k; k++) {
                        temp += a[i * size_k + k] * b[k * size_n + j];
                    }
                    temp *= alpha;
                    temp += beta * c[i * size_n + j];
                    c[i * size_n + j] = temp;
                }
            }
        } else if(!transA) {
            for(size_t i = 0; i < size_m; i++) {
                for(size_t j = 0; j < size_n; j++) {
                    temp = 0;
                    for(size_t k = 0; k < size_k; k++) {
                        temp += a[i * size_k + k] * b[j * size_k + k];
                    }
                    temp *= alpha;
                    temp += beta * c[i * size_n + j];
                    c[i * size_n + j] = temp;
                }
            }
        } else if(!transB) {
            for(size_t i = 0; i < size_m; i++) {
                for(size_t j = 0; j < size_n; j++) {
                    temp = 0;
                    for(size_t k = 0; k < size_k; k++) {
                        temp += a[k * size_m + i] * b[k * size_n + j];
                    }
                    temp *= alpha;
                    temp += beta * c[i * size_n + j];
                    c[i * size_n + j] = temp;
                }
            }
        } else {
            for(size_t i = 0; i < size_m; i++) {
                for(size_t j = 0; j < size_n; j++) {
                    temp = 0;
                    for(size_t k = 0; k < size_k; k++) {
                        temp += a[k * size_m + i] * b[j * size_k + k];
                    }
                    temp *= alpha;
                    temp += beta * c[i * size_n + j];
                    c[i * size_n + j] = temp;
                }
            }
        }

        return;  
    }
     
}

#endif