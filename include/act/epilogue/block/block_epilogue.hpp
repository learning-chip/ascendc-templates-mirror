/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
#define ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP

#include "act/act.hpp"

namespace Act::Epilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

}  // namespace Act::Epilogue::Block

#include "act/epilogue/block/block_epilogue_elemwise_one_source.hpp"
#include "act/epilogue/block/block_epilogue_fa_softmax.hpp"
#include "act/epilogue/block/block_epilogue_fa_rescal_o.hpp"
#include "act/epilogue/block/block_epilogue_mla_softmax.hpp"
#include "act/epilogue/block/block_epilogue_mla_rescal_o.hpp"
#include "act/epilogue/block/block_epilogue_mla_fd_rescal_o.hpp"
#include "act/epilogue/block/block_epilogue_per_token_dequant.hpp"
#include "act/epilogue/block/block_epilogue_gemm.hpp" 
#include "act/epilogue/block/block_epilogue_gemv.hpp" 
#include "act/epilogue/block/block_epilogue_mla_tp1_softmax.hpp"
#include "act/epilogue/block/block_epilogue_mla_tp1_rescal_o.hpp"
#endif  // ACT_EPILOGUE_BLOCK_BLOCK_EPILOGUE_HPP
