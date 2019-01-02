#include <stdexcept>
#include "deep500/deep500.h"
#include "../utility/math/gemm.hpp"
#include "../utility/tensor/broadcast.hpp"
#include "../utility/tensor/set_all.hpp"
#include "../utility/tensor/transpose2d.hpp"

template<typename T>
class MatMulLayer : public deep500::CustomOperator{

protected:
    /*
    if transA = 0 and trnsB = 0:
    shape of A = (M, K)
    shape of B = (K, N)
    shape of C should be unidirectional broadcastable to (M, N)
    */
    uint32_t m_M;
    uint32_t m_K;
    uint32_t m_N;

public: 
    //constructor
    MatMulLayer(
        const deep500::tensor_t a, 
        const deep500::tensor_t b){

        //assumes matrices, therefore either 1-d or 2-d tensors required
        if(a.dims == 2) {
            m_M = a.sizes[0];
            m_K = a.sizes[1];
        } else if(a.dims == 1) {
            m_M = a.sizes[0];
            m_K = 1;
        } else {
            throw std::runtime_error("gemmlayer: input tensor 'a' must eiher be of dimension 1 or 2");
        }

        if(b.dims == 2) {
            m_N = b.sizes[1];
        } else if(b.dims == 1) {
            m_N = 1;
        } else {
            throw std::runtime_error("gemmlayer: input (weight) tensor 2 'b' must eiher be of dimension 1 or 2");
        }
    }

    //copy constructor

    //destructor
    virtual ~MatMulLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, const T *input_tensor2, T *output_tensor) {
        
        /*
        ONNX notation:
        A = input
        B = input2
        */

        utilityTensor::set_all_zero(m_M * m_N, output_tensor);

        utilityMath::gemm(0, 0, m_M, m_N, m_K, 1, input_tensor, input_tensor2, 0., output_tensor);

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad) {

        //compute gradient with respect to the input
        const int transA_bwdinput = 0;
        const int transB_bwdinput = 1;
        const int m_bwdinput = m_M;
        const int k_bwdinput = m_N;
        const int n_bwdinput = m_K;
        const float alpha_bwdinput = 1;
        const float beta_bwdinput = 0.;

        utilityTensor::set_all_zero(m_M * m_K, input_tensor_grad);

        
        utilityMath::gemm(
            transA_bwdinput,
            transB_bwdinput,
            m_bwdinput,
            n_bwdinput,
            k_bwdinput,
            alpha_bwdinput,
            nextop_grad,
            fwd_input_tensor2,
            beta_bwdinput,
            input_tensor_grad);
        

        //compute gradient with respect to the weights
        //weight_grad = alpha * transpose(input) * nextop_grad
        const int transA_bwdinput2 = 1;
        const int transB_bwdinput2 = 0;
        const int m_bwdinput2 = m_K;
        const int k_bwdinput2 = m_M;
        const int n_bwdinput2 = m_N;
        const float alpha_bwdinput2 = 1;
        const float beta_bwdinput2 = 0.;

        utilityMath::gemm(transA_bwdinput2, transB_bwdinput2,
            m_bwdinput2, n_bwdinput2, k_bwdinput2,
            alpha_bwdinput2, fwd_input_tensor, nextop_grad, beta_bwdinput2, input_tensor2_grad);

        return;
    }
};

D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, 
                                    int num_inputs,
                                    deep500::tensor_t *output_descriptors,
                                    int num_outputs) {

    return new MatMulLayer<DTYPE>(input_descriptors[0], input_descriptors[1]);
}

D500_REGISTER_OP(MatMulLayer<DTYPE>);