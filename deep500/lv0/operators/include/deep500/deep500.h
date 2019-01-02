#pragma once

#include <cstdint>
#include <utility>

#ifdef _MSC_VER
    #define D500_EXPORTED extern "C" __declspec(dllexport)
#else
    #define D500_EXPORTED extern "C"
#endif

namespace deep500 {

//////////////////////////////////////////////////////////////////////////
// Data structures and types

    enum tensortype_t {
        TT_VOID = 0,
        TT_INT8,
        TT_INT16,
        TT_INT32,
        TT_INT64,
        TT_UINT8,
        TT_UINT16,
        TT_UINT32,
        TT_UINT64,
        TT_HALF,
        TT_FLOAT,
        TT_DOUBLE,
        TT_BITSET,

        // Number of elements in enum
        TT_LEN,
    };

    enum tensororder_t {
        TORD_IRRELEVANT = 0,
        TORD_NCHW,
        TORD_NHWC,
        TORD_LEN,
    };

    struct tensor_t
    {
        tensortype_t type;
        tensororder_t order;
        uint8_t dims;
        uint32_t *sizes;
    };

//////////////////////////////////////////////////////////////////////////
// Class that needs to be extended in order to implement a custom C++ operator
    class CustomOperator {
    public:
        CustomOperator() {}
        virtual ~CustomOperator() {}

        /** 
         * When extending this class, the functions forward(...) and 
         * backward(...) must be implemented (even if empty). The number of
         * arguments must match the operator descriptor (i.e., in `forward` as 
         * many pointer arguments as there are inputs, followed by the number
         * of output tensor argument pointers.
         * The data then points automatically with raw pointers, regardless of
         * the framework.
         *
         * For CUDA implementations (which receive GPU pointers), implement
         * forward_cuda(...) and backward_cuda(...) the same way.
         * The method `supports_cuda` must also return true.
         **/
        /*
        virtual void forward[_cuda](const Args... *input_tensors, 
                                      Args... *output_tensors) = 0;
        
        virtual void backward[_cuda](const Args... *nextop_grads,
                                     const Args... *fwd_input_tensors,
                                     const Args... *fwd_output_tensors,
                                     Args... *input_tensor_grads) = 0;
        */

        /// @return True if this operator contains implementations of CUDA 
        ///         functions. False otherwise.
        virtual bool supports_cuda() = 0;

        /// Reports information in the form of a custom data object, as well
        /// as a returned 64-bit integer. Example: bytes communicated thus far.
        virtual int64_t report(void *data) { return 0; }
    };

    
//////////////////////////////////////////////////////////////////////////
// Preprocessor macros to bind a void* and arguments to the actual operator
// methods.

    #define D500_REGISTER_OP_CPU(optype)                            \
    template<typename... Args>                                      \
    void _op_forward(void *handle, Args&&... args) {                \
        ((optype *)handle)->forward(std::forward<Args>(args)...);   \
    }                                                               \
    template<typename... Args>                                      \
    void _op_backward(void *handle, Args&&... args) {               \
        ((optype *)handle)->backward(                               \
            std::forward<Args>(args)...);                           \
    }

#ifdef __D500_OPHASCUDA
    #define D500_REGISTER_OP_GPU(optype)                            \
    template<typename... Args>                                      \
    void _op_forwardCuda(void *handle, Args&&... args) {            \
        ((optype *)handle)->forward_cuda(                           \
            std::forward<Args>(args)...);                           \
    }                                                               \
    template<typename... Args>                                      \
    void _op_backwardCuda(void *handle, Args&&... args) {           \
        ((optype *)handle)->backward_cuda(                          \
            std::forward<Args>(args)...);                           \
    }
#else
    #define D500_REGISTER_OP_GPU(optype)
#endif

    #define D500_REGISTER_OP(optype)                                \
    D500_REGISTER_OP_CPU(optype);                                   \
    D500_REGISTER_OP_GPU(optype)

}  // namespace deep500
