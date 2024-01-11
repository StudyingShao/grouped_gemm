/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/torch.h>
#include <cub/cub.cuh>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ATen/cuda/CUDAContext.h"

#include "cutlass_kernels/th_utils.h"
#include "cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include "cutlass_kernels/moe_gemm/moe_permute_kernels.h"
#include "cutlass_kernels/moe_gemm/moe_gemm_utils.h"
#include "cutlass_kernels/moe_gemm/moe_gemm_backward_template.h"

using torch::Tensor;

int64_t global_gemm_config_id = 0;

namespace torch_ext {

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Gemm Helper
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// act type, weight type
template <typename T, typename WeightType>
Tensor run_group_gemm_helper(Tensor input_activations,
                             Tensor expert_for_rows,
                             Tensor fc1_expert_weights,
                             const int num_experts)
{
    const int gemm_m = input_activations.size(0);
    const int gemm_n = fc1_expert_weights.size(2);
    const int gemm_k = input_activations.size(1);

    if (gemm_k & 0x7 != 0)
    {
        throw std::runtime_error("gemm_k must be a multiple of 8.");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    Tensor tokens_per_expert =
        torch::empty(num_experts, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int *tokens_per_expert_ptr = get_ptr<int>(tokens_per_expert);

    compute_rows_for_expert(
        expert_for_rows,
        tokens_per_expert_ptr,
        gemm_m,
        num_experts,
        stream);

    T *input_act_ptr = get_ptr<T>(input_activations);
    // int64_t *total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);
    WeightType *fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);

    const at::ScalarType _st = input_activations.scalar_type();
    auto fc1_output =
        torch::empty({gemm_m, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T *fc1_output_ptr = get_ptr<T>(fc1_output);

    groupedgemmformoe::MoeGemmRunner<T, WeightType> moe_gemm_runner_;

    moe_gemm_runner_.moe_gemm(input_act_ptr,
                              fc1_expert_weights_ptr,
                              fc1_output_ptr,
                              tokens_per_expert_ptr, // gemm_m
                              gemm_n,                // gemm_n
                              gemm_k,                // gemm_k
                              gemm_m,                // num_tokens
                              num_experts,
                              stream);

    return fc1_output;
}

// act type, weight type
template <typename T, typename WeightType>
Tensor run_group_gemm_backward_helper(Tensor input_activations,
                                      Tensor expert_for_rows,
                                      Tensor fc1_expert_weights,
                                      const int num_experts)
{
    // Matrix A: X      shape(m, k)
    // Matrix B: dL/dY  shape(m, n)
    // Output C: dL/dW  shape(k, n)

    const int gemm_m = input_activations.size(1);
    const int gemm_n = fc1_expert_weights.size(1);
    const int gemm_k = input_activations.size(0);

    if ((gemm_m & 0x7 != 0) || (gemm_n & 0x7 != 0))
    {
        throw std::runtime_error("gemm_k must be a multiple of 8.");
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    Tensor tokens_per_expert =
        torch::empty(num_experts, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    int *tokens_per_expert_ptr = get_ptr<int>(tokens_per_expert);

    compute_rows_for_expert(
        expert_for_rows,
        tokens_per_expert_ptr,
        gemm_k,
        num_experts,
        stream);

    T *input_act_ptr = get_ptr<T>(input_activations);
    WeightType *fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);

    const at::ScalarType _st = input_activations.scalar_type();
    auto fc1_output =
        torch::empty({num_experts, gemm_m, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    T *fc1_output_ptr = get_ptr<T>(fc1_output);

    groupedgemmformoe::MoeGemmRunner<T, WeightType> moe_gemm_runner_;

    moe_gemm_runner_.moe_gemm_backward(input_act_ptr,
                                       fc1_expert_weights_ptr,
                                       fc1_output_ptr,
                                       gemm_m,                // gemm_m
                                       gemm_n,                // gemm_n
                                       tokens_per_expert_ptr, // gemm_k
                                       gemm_k,                // num_tokens
                                       num_experts,
                                       stream);

    return fc1_output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Grouped GEMM OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

Tensor moe_group_gemm_op(Tensor input_activations,
                         Tensor expert_for_rows,
                         Tensor fc1_expert_weights,
                         int64_t num_experts)
{
    Tensor output_tensor;

    // activations type
    const at::ScalarType _st = input_activations.scalar_type();
    switch (_st) {
        case at::ScalarType::Float: {
            output_tensor = run_group_gemm_helper<float, float>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);
            break;
        }
        case at::ScalarType::Half: {
            output_tensor = run_group_gemm_helper<half, half>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            output_tensor = run_group_gemm_helper<__nv_bfloat16, __nv_bfloat16>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);
            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong activation tensor type.");
    }
    return output_tensor;
}

Tensor moe_group_gemm_backward_op(Tensor input_activations,
                                  Tensor expert_for_rows,
                                  Tensor fc1_expert_weights,
                                  int64_t num_experts)
{
    Tensor output_tensor;

    // activations type
    const at::ScalarType _st = input_activations.scalar_type();
    switch (_st) {
        case at::ScalarType::Float: {
            output_tensor = run_group_gemm_backward_helper<float, float>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);

            break;
        }
        case at::ScalarType::Half: {
            output_tensor = run_group_gemm_backward_helper<half, half>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);
            
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16: {
            output_tensor = run_group_gemm_backward_helper<__nv_bfloat16, __nv_bfloat16>(
                input_activations,
                expert_for_rows,
                fc1_expert_weights,
                num_experts);

            break;
        }
#endif
        default:
            throw std::runtime_error("Wrong activation tensor type.");
    }
    return output_tensor;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Permute OP
//
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, std::vector<Tensor>> moe_permute_op(
    Tensor original_input,
    Tensor expert_for_rows,
    std::vector<Tensor> workspace)
{
    const int num_rows = original_input.size(0);
    const int num_cols = original_input.size(1);

    if (original_input.is_cpu())
    {
        throw std::runtime_error("The input \"activations\" of permute op is on the device: CPU!.");
    }
    if (expert_for_rows.dtype() != torch::kInt32)
    {
        std::cerr << "Warning: The type of the input \"expert_for_rows\" of permute op is Int64! The recommended type is int32." << std::endl;
        expert_for_rows = expert_for_rows.to(torch::kInt32);
    }
    if (expert_for_rows.is_cpu())
    {
        std::cerr << "Warning: The input \"expert_for_rows\" of permute op is on the device: CPU!" << std::endl;
        expert_for_rows = expert_for_rows.to(torch::kCUDA);
    }

    // activations type
    const at::ScalarType _st = original_input.scalar_type();

    // initialize the workspace on the first run
    if (workspace.empty()) {
        // printf("Permute op workspace initialized!\n");

        auto options = torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(torch::kCUDA)
                        .requires_grad(false);
        Tensor row_id = torch::range(0, num_rows - 1, 1, options);
        Tensor sorted_expert_for_rows = torch::empty(num_rows, options);
        Tensor dest_row_to_source_row = torch::empty(num_rows, options);
        Tensor permuted_output =
            torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(row_id);
        workspace.push_back(sorted_expert_for_rows);
        workspace.push_back(dest_row_to_source_row);
        workspace.push_back(permuted_output);
    }

    int *expert_for_rows_ptr = get_ptr<int>(expert_for_rows);
    int *row_id_ptr = get_ptr<int>(workspace[0]);
    int *sorted_expert_for_rows_ptr = get_ptr<int>(workspace[1]);
    int *dest_row_to_source_row_ptr = get_ptr<int>(workspace[2]);
    Tensor &permuted_output = workspace[3];

    // Run sorting operation
    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    expert_for_rows_ptr, sorted_expert_for_rows_ptr,
                                    row_id_ptr, dest_row_to_source_row_ptr, num_rows);

    int *&map_dest_row_to_source_row = dest_row_to_source_row_ptr;
    Tensor &source_row_to_dest_row = workspace[1];
    int *&map_source_row_to_dest_row = sorted_expert_for_rows_ptr;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *original_input_ptr = get_ptr<dType>(original_input);
        dType *permuted_output_ptr = get_ptr<dType>(permuted_output);

        moe_permute_kernel_launcher<dType>(
            original_input_ptr,
            permuted_output_ptr,
            map_dest_row_to_source_row,
            map_source_row_to_dest_row,
            num_rows,
            num_cols,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return std::make_tuple(permuted_output, source_row_to_dest_row, workspace);
}

std::tuple<torch::Tensor, std::vector<Tensor>> moe_recover_op(
    Tensor permuted_input,
    Tensor source_row_to_dest_row,
    std::vector<Tensor> workspace)
{
    const int num_rows = permuted_input.size(0);
    const int num_cols = permuted_input.size(1);

    if (permuted_input.is_cpu())
    {
        throw std::runtime_error("The input \"permuted_input\" of permute op backward is on the device: CPU!.");
    }
    if (source_row_to_dest_row.is_cpu())
    {
        std::cerr << "Warning: The input \"source_row_to_dest_row\" of permute op backward is on the device: CPU!" << std::endl;
        source_row_to_dest_row = source_row_to_dest_row.to(torch::kCUDA);
    }

    // activations type
    const at::ScalarType _st = permuted_input.scalar_type();

    // initialize the workspace on the first run
    if (workspace.empty()) {
        // printf("Permute op backward workspace initialized!\n");

        Tensor original_output =
            torch::empty({num_rows, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(original_output);
    }

    Tensor &original_output = workspace[0];

    int *map_source_row_to_dest_row = get_ptr<int>(source_row_to_dest_row);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        using dType = float;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        using dType = half;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        using dType = __nv_bfloat16;

        dType *permuted_input_ptr = get_ptr<dType>(permuted_input);
        dType *original_output_ptr = get_ptr<dType>(original_output);

        moe_permute_kernel_launcher<dType>(
            permuted_input_ptr,
            original_output_ptr,
            map_source_row_to_dest_row,
            nullptr,
            num_rows,
            num_cols,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    cudaStreamSynchronize(stream);

    return std::make_tuple(original_output, workspace);
}

void moe_set_gemm_config(int64_t gemm_config_id)
{
    global_gemm_config_id = gemm_config_id;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// TORCH_LIBRARY
//
/////////////////////////////////////////////////////////////////////////////////////////////////

TORCH_LIBRARY(moe_unit_ops, m)
{
    m.def("moe_group_gemm_op", moe_group_gemm_op);
    m.def("moe_group_gemm_backward_op", moe_group_gemm_backward_op);
    m.def("moe_permute_op", moe_permute_op);
    m.def("moe_recover_op", moe_recover_op);
    m.def("moe_set_gemm_config", moe_set_gemm_config);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
}