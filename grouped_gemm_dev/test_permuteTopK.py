# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import triton
import torch.cuda.nvtx as nvtx

try:
  from grouped_gemm import permute_topK, unpermute_topK
except ImportError:
  print("grouped-gemm toolkit is not installed. Fall back to local import.")
  # For local debug
  from moe.ops import permute_topK, unpermute_topK

def permute(tokens, indices, expand_factor: int = 1):
    """Permute the tokens based on the indices.

    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): The token2expert indices tensor.

    Returns:
        torch.Tensor: The permuted tensor.
    """
    expand_factor = indices.size(1)

    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def unpermute(permuted_tokens, sorted_indices, probs: torch.Tensor = None, merge_factor: int = 1):
    """Unpermute the sorted tokens based on the indices.
    
    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The sorted indices tensor.
        probs (torch.Tensor, optional): The probabilities tensor. Defaults to None.
        merge_factor (int, optional): The merge factor. Defaults to 1.

    Returns:
        torch.Tensor: The unpermuted tensor.
    """
    merge_factor = probs.size(1)

    if merge_factor > 1:
        assert probs is not None
        assert (
            probs.size(0) == permuted_tokens.size(0) // merge_factor
        ), f"{probs.size()} {permuted_tokens.size()}"
    if probs is not None:
        assert probs.size(0) == permuted_tokens.size(0) // merge_factor
        assert (
            probs.size(1) == merge_factor
        ), f"probs size {probs.size()} merge_factor {merge_factor}"

    # unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens = permuted_tokens.index_copy(0, sorted_indices, permuted_tokens)

    unpermuted_tokens = unpermuted_tokens.reshape(-1, merge_factor, permuted_tokens.size(-1))

    if probs is not None:
        dtype = unpermuted_tokens.dtype
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.to(dtype)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens

def permute_topK_test(
    dtype,
    num_token,
    num_expert,
    hidden_size,
    num_topK,
    PRINT,
    BENCHMARK):
    
    print(f"{dtype} token:{num_token} hidden_size:{hidden_size} expert:{num_expert} topK:{num_topK}")

    input = torch.rand((num_token, hidden_size), dtype=dtype)
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     input[i][j] = i * 100 + j
    
    indices = torch.stack([torch.randperm(num_expert)[:num_topK] for _ in range(num_token)])


    # probs = torch.tensor([[0.1, 0.9],
    #                       [0.2, 0.8],
    #                       [0.3, 0.7]])
    # 0.5
    # probs = torch.ones_like(indices) / 2
    # rand
    probs = torch.rand(num_token, num_topK)
    row_sums = probs.sum(dim=1, keepdim=True)
    probs = probs / row_sums

    if PRINT:
        print(input)
        print(indices)
        print(probs)

    input = input.cuda()
    input_fp32 = input.detach().to(torch.float32)
    input_ = input.detach()
    indices = indices.to(torch.int32).cuda()
    probs = probs.cuda()
    input.requires_grad_(True)
    input_fp32.requires_grad_(True)
    input_.requires_grad_(True)
    probs.requires_grad_(True)

    ###################################################################################################################################
    #
    # PyTorch
    #
    ###################################################################################################################################
    permuted_tokens_fp32, sorted_indices = permute(input_fp32, indices, num_topK)
    permuted_tokens, sorted_indices = permute(input, indices, num_topK)
    # print("-----------------------------------------------------------------")
    # print(permuted_tokens)
    # print(sorted_indices)

    backward_input = torch.rand_like(permuted_tokens)
    backward_input_fp32 = backward_input.detach().to(torch.float32)
    # for i in range(num_token * num_topK):
    #   for j in range(hidden_size):
    #     backward_input[i][j] = i * 100 + j
    # print(backward_input)

    permuted_tokens_fp32.backward(backward_input_fp32, retain_graph=True)
    permuted_tokens.backward(backward_input, retain_graph=True)

    unpermuted_tokens = unpermute(
        permuted_tokens, sorted_indices, probs=probs, merge_factor=num_topK)

    if PRINT:
        print("--------------unpermute fwd input--------------")
        print(permuted_tokens)
        print("--------------unpermute fwd output--------------")
        print(unpermuted_tokens)

    backward_input_unperm = torch.rand_like(unpermuted_tokens)
    # for i in range(num_token):
    #   for j in range(hidden_size):
    #     backward_input_unperm[i][j] = i * 2000 + j * 20

    if PRINT:
        print("--------------unpermute bwd input--------------")
        print(backward_input_unperm)

    permuted_tokens.retain_grad()
    permuted_tokens.grad = torch.zeros_like(permuted_tokens)

    unpermuted_tokens.backward(backward_input_unperm, retain_graph=True)
    if PRINT:
        print("--------------unpermute bwd output act grad--------------")
        print(permuted_tokens.grad)
        print("--------------unpermute bwd output probs grad--------------")
        print(probs.grad)

    ###################################################################################################################################
    #
    # Mine
    #
    ###################################################################################################################################
    permuted_act, row_id_map = permute_topK(input_, indices)
    assert torch.allclose(permuted_tokens, permuted_act)

    if PRINT:
        print("--------------row_id_map--------------")
        print(row_id_map)
        print("--------------input_--------------")
        print(input_)
        print("--------------permuted_act--------------")
        print(permuted_act)

    permuted_act.backward(backward_input, retain_graph=True)

    if torch.allclose(input.grad, input_.grad) == False:
        original_inputs = input_.grad.float().cpu().numpy().flatten()
        original_output = input_fp32.grad.float().cpu().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"permute_topK bwd max error (mine vs FP32): \t\t\t{max_abs_error:.3e} ({dtype})")

        original_inputs = input.grad.float().cpu().numpy().flatten()
        original_output = input_fp32.grad.float().cpu().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"permute_topK bwd max error (pytorch vs FP32): \t\t\t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(input.grad)
            print(input_.grad)
            print(input_fp32.grad)

    probs_mine = probs.detach().clone()
    probs_mine.requires_grad_(True)

    unpermuted_act = unpermute_topK(permuted_act, row_id_map, probs_mine)

    if torch.allclose(unpermuted_tokens, unpermuted_act) == False:
        original_inputs = unpermuted_tokens.float().cpu().detach().numpy().flatten()
        original_output = unpermuted_act.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK fwd max error (mine vs pytorch): \t\t{max_abs_error:.3e} ({dtype})")
        
        if PRINT:
            print(unpermuted_tokens)
            print(unpermuted_act)

    permuted_act.retain_grad()
    probs_mine.retain_grad()

    unpermuted_act.backward(backward_input_unperm, retain_graph=True)
    if torch.allclose(permuted_tokens.grad, permuted_act.grad):
        original_inputs = permuted_tokens.grad.float().cpu().detach().numpy().flatten()
        original_output = permuted_act.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd act_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(backward_input_unperm)
            print(permuted_act.grad)
            print(permuted_tokens.grad)
            print(probs_mine.grad)
            print(probs.grad)

    if torch.allclose(probs_mine.grad, probs.grad) == False:
        original_inputs = probs_mine.grad.float().cpu().detach().numpy().flatten()
        original_output = probs.grad.float().cpu().detach().numpy().flatten()
        max_abs_error = abs(original_inputs - original_output).max()
        print(f"unpermute_topK bwd prob_grad max error (mine vs pytorch): \t{max_abs_error:.3e} ({dtype})")
        if PRINT:
            print(probs_mine.grad)
            print(probs.grad)

    ###################################################################################################################################
    #
    # Benchmark
    #
    ###################################################################################################################################
    if BENCHMARK:
        print(f"----permute topK----")
        t1 = triton.testing.do_bench(lambda: permute(input, indices, 2))
        print(f"pytorch fwd: {t1:.3f} ms")
        t2 = triton.testing.do_bench(lambda: permuted_tokens.backward(backward_input, retain_graph=True))
        print(f"pytorch bwd: {t2:.3f} ms")

        t3 = triton.testing.do_bench(lambda: permute_topK(input_, indices))
        print(f"mine    fwd: {t3:.3f} ms")
        t4 = triton.testing.do_bench(
            lambda: permuted_act.backward(backward_input, retain_graph=True))
        print(f"mine    bwd: {t4:.3f} ms")

        print(f"----unpermute topK----")
        t3 = triton.testing.do_bench(
            lambda: unpermute(permuted_tokens, sorted_indices, probs=probs, merge_factor=num_topK))
        print(f"pytorch fwd: {t3:.3f} ms")
        t4 = triton.testing.do_bench(
            lambda: unpermuted_tokens.backward(backward_input_unperm, retain_graph=True))
        print(f"pytorch bwd: {t4:.3f} ms")

        t3 = triton.testing.do_bench(
            lambda: unpermute_topK(permuted_act, row_id_map, probs_mine))
        print(f"mine    fwd: {t3:.3f} ms")
        t4 = triton.testing.do_bench(
            lambda: unpermuted_act.backward(backward_input_unperm, retain_graph=True))
        print(f"mine    bwd: {t4:.3f} ms")

    # perf_test_cuda_kernel(lambda: permute(input, indices, 2))
    # perf_test_cuda_kernel(lambda: permuted_tokens.backward(
    #     backward_input, retain_graph=True))
    # perf_test_cuda_kernel(lambda: permute_topK(input_, indices))
    # perf_test_cuda_kernel(lambda: permuted_act.backward(
    #     backward_input, retain_graph=True))

def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        # create CUDA event
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(20):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Elapsed Time: {elapsed_time_ms / 20} ms")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":

    torch.manual_seed(1)

    # num_token = 4
    # num_expert = 3
    # hidden_size = 8
    # num_topK = 2

    num_token = 4096
    num_expert = 4
    hidden_size = 4096
    num_topK = 2

    Benchmark = False

    dtype = torch.float32
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.float16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, num_expert,
                      hidden_size, num_topK, False, Benchmark)
    dtype = torch.bfloat16
    permute_topK_test(dtype, num_token, 4, hidden_size, 1, False, Benchmark)
    permute_topK_test(dtype, num_token, 5, hidden_size, 2, False, Benchmark)
    permute_topK_test(dtype, num_token, 6, hidden_size, 3, False, Benchmark)
    permute_topK_test(dtype, num_token, 7, hidden_size, 4, False, Benchmark)
    permute_topK_test(dtype, num_token, 8, hidden_size, 5, False, Benchmark)