import torch
import triton
import torch.cuda.nvtx as nvtx

# from grouped_gemm import permute, unpermute, groupedgemm
from moe.ops import permute, unpermute, groupedgemm


moe_set_gemm_config = torch.ops.moe_unit_ops.moe_set_gemm_config

class GEMM_CASE:
  # M shards, M, N, K
  def __init__(self, num_experts, num_tokens, inter, hidden):
    self.num_experts = num_experts
    self.num_tokens = num_tokens
    self.inter = inter
    self.hidden = hidden

if __name__ == "__main__":
  
  dtype = torch.bfloat16

  gemm_cases = []

  gemm_cases.append(GEMM_CASE(8, 4096*2, 2048*4, 2048))
  gemm_cases.append(GEMM_CASE(8, 4096*2, 2048, 2048*4))

  gemm_cases.append(GEMM_CASE(64, 32768, 2048*4, 2048))
  gemm_cases.append(GEMM_CASE(64, 32768, 2048, 2048*4))

  for groups in [2,4,6,8,16]:
    for M in [2048, 4096]:
      gemm_cases.append(GEMM_CASE(groups, M, 2048, 768))
      gemm_cases.append(GEMM_CASE(groups, M, 768, 2048))
      gemm_cases.append(GEMM_CASE(groups, M, 2816, 1024))
      gemm_cases.append(GEMM_CASE(groups, M, 1024, 2816))
  
  for groups in [1,2,4,8]:
    for M in [8192, 16384]:
      gemm_cases.append(GEMM_CASE(groups, M, 14336, 4096))
      gemm_cases.append(GEMM_CASE(groups, M, 4096, 14336))
    for M in [16384, 32768]:
      gemm_cases.append(GEMM_CASE(groups, M, 30720, 15360))
      gemm_cases.append(GEMM_CASE(groups, M, 15360, 30720))
      gemm_cases.append(GEMM_CASE(groups, M, 14336, 4096))
      gemm_cases.append(GEMM_CASE(groups, M, 4096, 14336))

  for gemm_case in gemm_cases:

    num_experts = gemm_case.num_experts
    num_tokens = gemm_case.num_tokens
    inter = gemm_case.inter
    hidden = gemm_case.hidden

    flops = num_tokens * hidden * inter * 2

    for config_id in range(1, 6):
      print(f"{gemm_case.num_experts}/{gemm_case.num_tokens}/{gemm_case.inter}/{gemm_case.hidden}  ", end='')
      print(f"{config_id}  ", end='')
      for stage_id in range(2, 11):
        moe_set_gemm_config(config_id, stage_id)

        expert_for_rows = torch.randint(size=(num_tokens,), low=0, high=num_experts, dtype=torch.int32).cuda()
        for i in range(num_experts):
          for j in range(num_tokens // num_experts):
            tid = i * num_tokens // num_experts + j
            expert_for_rows[tid] = i

        permuted_inputs = torch.randn([num_tokens, hidden], dtype=dtype, device="cuda")
        weights = torch.randn([num_experts, hidden, inter], dtype=dtype, device="cuda")
        
        nvtx.range_push("warmups")
        for _ in range(50):
          gemm_output = groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts)
        nvtx.range_pop()
  
        t = triton.testing.do_bench(lambda: groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts))
        tflops = flops / 1e12 / (t / 1e3)
        if tflops < 1000:
          print(f"{tflops:.2f} ", end='')
        else:
          print("- ", end='')

      print()

    # permuted_inputs = torch.randn([num_tokens, hidden], dtype=dtype, device="cuda")
    # weights = torch.randn([num_experts, hidden, inter], dtype=dtype, device="cuda")
    # permuted_inputs = permuted_inputs.view([num_experts, num_tokens // num_experts, hidden])

    # t = triton.testing.do_bench(lambda: torch.bmm(permuted_inputs, weights))
    # tflops = flops / 1e12 / (t / 1e3)
    # print(f"Ref torch.bmm Perf: {tflops:.2f} TFLOPS, {tflops/312*100:.2f} %")





