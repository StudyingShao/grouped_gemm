/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"

template <typename T, int kElementsPerAccess>
__global__ void moe_permute_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *row_id_map,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int dest_row = blockIdx.x;

    if (dest_row >= num_rows)
        return;

    int source_row = row_id_map[dest_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}

template <typename T, int kElementsPerAccess>
__global__ void moe_recover_kernel(const T *original_input,
                                   T *permuted_output,
                                   const int *row_id_map,
                                   const int num_rows,
                                   const int num_cols)
{
    // Reverse permutation map.
    // each block corresponds to one row
    const int source_row = blockIdx.x;

    if (source_row >= num_rows)
        return;

    int dest_row = row_id_map[source_row];

    // permute activations rows based on experts
    const T *source_row_ptr = original_input + source_row * num_cols;
    T *dest_row_ptr = permuted_output + dest_row * num_cols;

    for (int tid = threadIdx.x * kElementsPerAccess; tid < num_cols; tid += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<float4, sizeof(float4), cutlass::arch::CacheOperation::LastUse>(
            *(float4 *)(dest_row_ptr + tid), (source_row_ptr + tid), true);
    }
}

template <typename T, bool FWD, int kElementsPerAccess>
void moe_permute_kernel_launcher(
    const T *original_input,
    T *permuted_output,
    const int *row_id_map,
    const int num_rows,
    const int num_cols,
    cudaStream_t stream)
{
    if (num_cols & 0x7 != 0)
        throw std::runtime_error("num_cols of input activations must be multiples of 8.");

    const int blocks = num_rows;
    const int threads = std::min(num_cols / kElementsPerAccess, 1024);

    if (FWD)
    {
        moe_permute_kernel<T, kElementsPerAccess><<<blocks, threads, 0, stream>>>(original_input,
                                                                                  permuted_output,
                                                                                  row_id_map,
                                                                                  num_rows,
                                                                                  num_cols);
    }
    else
    {
        moe_recover_kernel<T, kElementsPerAccess><<<blocks, threads, 0, stream>>>(original_input,
                                                                                  permuted_output,
                                                                                  row_id_map,
                                                                                  num_rows,
                                                                                  num_cols);
    }
}