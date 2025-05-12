# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch_npu
import torch
import torch_act
import numpy as np
torch.ops.load_library("./output/python_extension/libact_torch.so") 


device = "npu:0"

torch.manual_seed(0)

M = N = K = 32

low = -16
high = 16
batch_size = 12

a_int = torch.randint(low=low, high=high, size = (batch_size, M, K), device=device).to(torch.int8)
b_int = torch.randint(low=low, high=high, size = (batch_size, K, N), device=device).to(torch.int8)
bqmm_result = torch.zeros((batch_size, M, N), device=device).to(torch.float16) # bqmm -> Batched Quant Matmul

scale_bf16 = torch.ones(N, dtype=torch.bfloat16, device=device)
per_token_scale = torch.ones(M, dtype=torch.bfloat16, device=device)

a_float = a_int.to(torch.float16)
b_float = b_int.to(torch.float16)

torch_baseline = torch.bmm(a_float, b_float)
torch_act.batched_quant_matmul(a_int, b_int, bqmm_result, "float16", np.float32(1.0))
qmm_result = torch_act.quant_matmul(a_int[0], b_int[0], scale_bf16, per_token_scale, "bf16")


is_qmm_correct = torch.allclose(torch_baseline[0].to(torch.bfloat16), qmm_result, atol=1)
is_bqmm_correct = torch.allclose(torch_baseline, bqmm_result, atol=1)

if is_qmm_correct and is_bqmm_correct:
    print("Both torch_act.batched_quant_matmul and torch_act.quant_matmul matched with baseline!")

if is_qmm_correct == False:
    print("torch_act.quant_matmul did not match the results of torch baseline!")

if is_bqmm_correct == False:
    print("torch_act.batched_quant_matmul did not match the results of torch baseline!")

