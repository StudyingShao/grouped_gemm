# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

torch.classes.load_library("./build/libmoe_unit_ops.so")

# TODO by Jiang Shao, add parameter `out` which can be optionally given to be used as output buffers.

class PermuteMoE(torch.autograd.Function):
  
  workspace_fw=None
  workspace_bw=None
  dtype=None
  num_rows=None

  @staticmethod
  def forward(ctx, unpermuted_inputs, expert_for_rows):

    if PermuteMoE.num_rows != unpermuted_inputs.size(0) or PermuteMoE.dtype != unpermuted_inputs.dtype:
      # print("Permute op workspace reset!")
      PermuteMoE.num_rows = unpermuted_inputs.size(0)
      PermuteMoE.dtype = unpermuted_inputs.dtype
      PermuteMoE.workspace_fw = []
      PermuteMoE.workspace_bw = []

    permuted_inputs, source_row_to_dest_row, PermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs,
      expert_for_rows,
      PermuteMoE.workspace_fw)

    ctx.source_row_to_dest_row = source_row_to_dest_row

    return permuted_inputs, source_row_to_dest_row

  @staticmethod
  def backward(ctx, permuted_inputs_grad, _):
    permuted_inputs_grad = permuted_inputs_grad.contiguous()
    source_row_to_dest_row = ctx.source_row_to_dest_row

    original_output, PermuteMoE.workspace_bw = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs_grad,
      source_row_to_dest_row,
      PermuteMoE.workspace_bw)
    return original_output, None


class UnpermuteMoE(torch.autograd.Function):

  workspace_fw=None
  workspace_bw=None
  dtype=None
  num_rows=None
  
  @staticmethod
  def forward(ctx, permuted_inputs, expert_for_rows, source_row_to_dest_row):
    
    if UnpermuteMoE.num_rows != permuted_inputs.size(0) or UnpermuteMoE.dtype != permuted_inputs.dtype:
      # print("Unpermute op workspace reset!")
      UnpermuteMoE.num_rows = permuted_inputs.size(0)
      UnpermuteMoE.dtype = permuted_inputs.dtype
      UnpermuteMoE.workspace_fw = []
      UnpermuteMoE.workspace_bw = []

    ctx.expert_for_rows = expert_for_rows

    original_output, UnpermuteMoE.workspace_bw = torch.ops.moe_unit_ops.moe_recover_op(
      permuted_inputs,
      source_row_to_dest_row,
      UnpermuteMoE.workspace_bw)
    
    return original_output
  
  @staticmethod
  def backward(ctx, unpermuted_inputs_grad):

    unpermuted_inputs_grad = unpermuted_inputs_grad.contiguous()
    expert_for_rows = ctx.expert_for_rows

    permuted_inputs, _, UnpermuteMoE.workspace_fw = torch.ops.moe_unit_ops.moe_permute_op(
      unpermuted_inputs_grad,
      expert_for_rows,
      UnpermuteMoE.workspace_fw)

    return permuted_inputs, None, None


class GroupedGemmMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, permuted_inputs, expert_for_rows, weights, num_experts):
      output = torch.ops.moe_unit_ops.moe_group_gemm_op(
        permuted_inputs,
        expert_for_rows,
        weights,
        num_experts)
      
      ctx.save_for_backward(permuted_inputs, expert_for_rows, weights)
      ctx.num_experts = num_experts

      return output


    @staticmethod
    def backward(ctx, permuted_inputs_grad):
        
      permuted_inputs, expert_for_rows, weights = ctx.saved_tensors
      num_experts = ctx.num_experts
      permuted_inputs_grad = permuted_inputs_grad.contiguous()

      weight_grad = None
      if ctx.needs_input_grad[0]:
        weight_grad = torch.ops.moe_unit_ops.moe_group_gemm_backward_op(
          permuted_inputs,
          expert_for_rows,
          permuted_inputs_grad,
          num_experts)

      activation_grad = None
      if ctx.needs_input_grad[2]:
        # TODO by Jiang Shao, add trans_b to avoid weight permutation
        activation_grad = torch.ops.moe_unit_ops.moe_group_gemm_op(
          permuted_inputs_grad,
          expert_for_rows,
          weights.permute(0, 2, 1).contiguous(),
          num_experts)

      return activation_grad, None, weight_grad, None


def permute(unpermuted_inputs, expert_for_rows):
  return PermuteMoE.apply(unpermuted_inputs, expert_for_rows)

def unpermute(permuted_inputs, expert_for_rows, source_row_to_dest_row):
  return UnpermuteMoE.apply(permuted_inputs, expert_for_rows, source_row_to_dest_row)

def groupedgemm(permuted_inputs, expert_for_rows, weights, num_experts):
  return GroupedGemmMoE.apply(permuted_inputs, expert_for_rows, weights, num_experts)