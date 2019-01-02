#include <stdexcept>
#include "deep500/deep500.h"
#include "../utility/math/gemm.hpp"
#include "../utility/tensor/broadcast.hpp"
#include "../utility/tensor/set_all.hpp"
#include "../utility/tensor/transpose2d.hpp"

template<typename T>
class GemmLayer : public deep500::CustomOperator{

protected:
    float m_alpha;
    float m_beta;
    int m_transA;
    int m_transB;

    /*
    if transA = 0 and trnsB = 0:
    shape of A = (M, K)
    shape of B = (K, N)
    shape of C should be unidirectional broadcastable to (M, N)
    */
    uint32_t m_M;
    uint32_t m_K;
    uint32_t m_N;
    uint32_t m_Mc;
    uint32_t m_Nc;

public: 
    //constructor
    GemmLayer(
        const deep500::tensor_t a, 
        const deep500::tensor_t b,
        const deep500::tensor_t c,
        const float alpha,
        const float beta,
        const int transA,
        const int transB) :
        m_alpha(alpha),
        m_beta(beta),
        m_transA(transA),
        m_transB(transB){

        //assumes matrices, therefore either 1-d or 2-d tensors required
        if(a.dims == 2) {
            if(!transA) {
                m_M = a.sizes[0];
                m_K = a.sizes[1];
            } else {
                m_M = a.sizes[1];
                m_K = a.sizes[0];
            }
        } else if(a.dims == 1) {
            if(!transA) {
                m_M = a.sizes[0];
                m_K = 1;
            } else {
                m_M = 1;
                m_K = a.sizes[0];
            }
        } else {
            throw std::runtime_error("gemmlayer: input tensor 'a' must eiher be of dimension 1 or 2");
        }

        if(b.dims == 2) {
            if(!transB) {
                m_N = b.sizes[1];
            } else {
                m_N = b.sizes[0];
            }
        } else if(b.dims == 1) {
            if(!transB) {
                m_N = 1;
            } else {
                m_N = b.sizes[0];
            }
        } else {
            throw std::runtime_error("gemmlayer: input (weight) tensor 2 'b' must eiher be of dimension 1 or 2");
        }

        if(c.dims == 2) {
            m_Mc = c.sizes[0];
            m_Nc = c.sizes[1];
        } else if(c.dims == 1) {
            m_Mc = c.sizes[0];
            m_Nc = 1;
        } else {
            throw std::runtime_error("gemmlayer: input (bias) tensor 3 'c' must eiher be of dimension 1 or 2");
        }
    }

    //copy constructor

    //destructor
    virtual ~GemmLayer() {}

    //member functions
    virtual bool supports_cuda() { return false; }

    void forward(const T *input_tensor, const T *input_tensor2, const T *input_tensor3, T *output_tensor) {
        
        /*
        ONNX notation:
        A = input
        B = input2
        C = input3
        */

        utilityTensor::unidir_broadcast2d(m_M, m_N, m_Mc, m_Nc, input_tensor3, output_tensor);

        utilityMath::gemm(m_transA, m_transB, m_M, m_N, m_K, m_alpha, input_tensor, input_tensor2, m_beta, output_tensor);

        return;
    }

    void backward(
        const T *nextop_grad,
        const T *fwd_input_tensor,
        const T *fwd_input_tensor2,
        const T *fwd_input_tensor3,
        const T *fwd_output_tensor,
        T *input_tensor_grad,
        T *input_tensor2_grad,
        T *input_tensor3_grad) {

        //compute gradient with respect to the input
        const int transA_bwdinput = m_transA;
        const int transB_bwdinput = m_transA? m_transB : !m_transB;
        const int m_bwdinput = m_M;
        const int k_bwdinput = m_N;
        const int n_bwdinput = m_K;
        const float alpha_bwdinput = m_alpha;
        const float beta_bwdinput = 0.;

        utilityTensor::set_all_zero(m_M * m_K, input_tensor_grad);

        if(m_transA == 0) {
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
        }else {
            utilityMath::gemm(
                transB_bwdinput,
                transA_bwdinput,
                n_bwdinput,
                m_bwdinput,
                k_bwdinput,
                alpha_bwdinput,
                fwd_input_tensor2,
                nextop_grad,
                beta_bwdinput,
                input_tensor_grad);
        }

        //compute gradient with respect to the weights
        //weight_grad = alpha * transpose(input) * nextop_grad
        const int transA_bwdinput2 = !m_transA;
        const int transB_bwdinput2 = 0;
        const int m_bwdinput2 = m_K;
        const int k_bwdinput2 = m_M;
        const int n_bwdinput2 = m_N;
        const float alpha_bwdinput2 = m_alpha;
        const float beta_bwdinput2 = 0.;

        utilityMath::gemm(transA_bwdinput2, transB_bwdinput2,
            m_bwdinput2, n_bwdinput2, k_bwdinput2,
            alpha_bwdinput2, fwd_input_tensor, nextop_grad, beta_bwdinput2, input_tensor2_grad);

        if(m_transB) {
            utilityTensor::transpose2d(m_bwdinput2, n_bwdinput2, input_tensor2_grad);
        }

        //compute gradient with respect to the bias
        //bias_grad = beta * nextop_grad
        if(m_M == m_Mc && m_N == m_Nc) {
            for(size_t i = 0; i < m_M * m_N; i++) {
                input_tensor3_grad[i] = nextop_grad[i] * m_beta;
            }
        }else if(m_M == m_Mc) {
            utilityTensor::set_all_zero(m_M, input_tensor3_grad);
            for(size_t i = 0; i < m_M; i++) {
                for(size_t j = 0; j < m_N; j++) {
                    input_tensor3_grad[i] += nextop_grad[i * m_N + j] * m_beta;
                }
            }
        }else if(m_N == m_Nc) {
            utilityTensor::set_all_zero(m_N, input_tensor3_grad);
            for(size_t i = 0; i < m_M; i++) {
                for(size_t j = 0; j < m_N; j++) {
                    input_tensor3_grad[j] += nextop_grad[i * m_N + j] * m_beta;
                }
            }
        }else{
            input_tensor3_grad[0] = 0;
            for(size_t i = 0; i < m_M; i++) {
                for(size_t j = 0; j < m_N; j++) {
                    input_tensor3_grad[0] += nextop_grad[i * m_N + j];
                }
            }
            input_tensor3_grad[0] *= m_beta;
        } 

        return;
    }
};

D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, 
                                    int num_inputs,
                                    deep500::tensor_t *output_descriptors,
                                    int num_outputs) {

    return new GemmLayer<DTYPE>(input_descriptors[0], input_descriptors[1], input_descriptors[2], ALPHA, BETA, TRANSA, TRANSB);
}

D500_REGISTER_OP(GemmLayer<DTYPE>);