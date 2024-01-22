/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <cuda_runtime_api.h>

#include "moe/cutlass_kernels/th_utils.h"
#include "ft_gemm_configs.h"

extern int64_t global_gemm_config_id;
extern int64_t global_gemm_stage_id;

namespace groupedgemmformoe {

template<typename T, /*The type used for activations/scales/compute*/
         typename WeightType /* The type for the MoE weights */>
class MoeGemmRunner {
public:
    MoeGemmRunner()
    {
        int device{-1};
        check_cuda_error(cudaGetDevice(&device));
        sm_ = getSMVersion();
        check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
    }

    void moe_gemm(T*           A,
                  WeightType*  B,
                  T*           C,
                  int*         gemm_m_per_expert,
                  int64_t      gemm_n,
                  int64_t      gemm_k,
                  int          num_tokens,
                  int          num_experts,
                  cudaStream_t stream);

    void moe_gemm_backward(T*           A,
                           WeightType*  B,
                           T*           C,
                           int64_t      gemm_m,
                           int64_t      gemm_n,
                           int*         gemm_k_per_expert,
                           int          num_tokens,
                           int          num_experts,
                           cudaStream_t stream);

private:
    void dispatch_to_arch(T*                A,
                          WeightType*       B,
                          T*                C,
                          int*              gemm_m_per_expert,
                          int64_t           gemm_n,
                          int64_t           gemm_k,
                          int               num_experts,
                          CutlassGemmConfig gemm_config,
                          cudaStream_t      stream,
                          int*              occupancy = nullptr);

    void run_gemm(T*           A,
                  WeightType*  B,
                  T*           C,
                  int*         gemm_m_per_expert,
                  int64_t      gemm_n,
                  int64_t      gemm_k,
                  int          num_tokens,
                  int          num_experts,
                  cudaStream_t stream);

    void dispatch_to_arch_backward(T*                A,
                                   WeightType*       B,
                                   T*                C,
                                   int64_t           gemm_m,
                                   int64_t           gemm_n,
                                   int*              gemm_k_per_expert,
                                   int               num_experts,
                                   CutlassGemmConfig gemm_config,
                                   cudaStream_t      stream,
                                   int*              occupancy = nullptr);

    void run_gemm_backward(T*           A,
                           WeightType*  B,
                           T*           C,
                           int64_t      gemm_m,
                           int64_t      gemm_n,
                           int*         gemm_k_per_expert,
                           int          num_tokens,
                           int          num_experts,
                           cudaStream_t stream);

private:
    int sm_;
    int multi_processor_count_;
};

}  // namespace groupedgemmformoe