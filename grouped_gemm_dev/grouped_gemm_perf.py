import torch
import triton
import torch.cuda.nvtx as nvtx

# from grouped_gemm import permute, unpermute, groupedgemm
from moe.ops import permute, unpermute, groupedgemm


moe_set_gemm_config = torch.ops.moe_unit_ops.moe_set_gemm_config

if __name__ == "__main__":
  for test in range(1, 6):
    dtype = torch.bfloat16
    # num_experts = 8
    # num_tokens = 4096 * 2
    # hidden = 2048
    # inter = 2048 * 4

    num_experts = 64
    num_tokens = 32768
    hidden = 2048
    inter = 2048 * 4
    
    # num_experts = 64
    # num_tokens = 32768
    # hidden = 2048 * 4
    # inter = 2048 * 4

    # num_experts = 128
    # num_tokens = 128 * 16
    # hidden = 2048

    print("config: ", test)
    moe_set_gemm_config(test)

    expert_for_rows = torch.randint(size=(num_tokens,), low=0, high=num_experts, dtype=torch.int32).cuda()
    for i in range(num_experts):
      for j in range(num_tokens // num_experts):
        tid = i * num_tokens // num_experts + j
        expert_for_rows[tid] = i

    permuted_inputs = torch.randn([num_tokens, hidden], dtype=dtype, device="cuda")
    weights = torch.randn([num_experts, hidden, inter], dtype=dtype, device="cuda")
    
    nvtx.range_push("groupedgemm")
    for _ in range(50):
      gemm_output = groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts)
    nvtx.range_pop()

    flops = num_tokens * hidden * inter * 2
    t = triton.testing.do_bench(lambda: groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts))
    
    tflops = flops / 1e12 / (t / 1e3)
    print(f"Perf: {tflops:.2f} TFLOPS, {tflops/312*100:.2f} %")

    nvtx.range_push("groupedgemm")
    gemm_output = groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts)
    nvtx.range_pop()

    permuted_inputs = permuted_inputs.view([num_experts, num_tokens // num_experts, hidden])
    torch.bmm(permuted_inputs, weights)

    t = triton.testing.do_bench(lambda: torch.bmm(permuted_inputs, weights))
    tflops = flops / 1e12 / (t / 1e3)
    print(f"Perf: {tflops:.2f} TFLOPS, {tflops/312*100:.2f} %")

