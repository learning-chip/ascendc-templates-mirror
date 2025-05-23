/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACT_GEMM_BLOCK_BLOCK_MMAD_HPP
#define ACT_GEMM_BLOCK_BLOCK_MMAD_HPP

#include "act/act.hpp"
#include "act/gemm/tile/tile_copy.hpp"
#include "act/gemm/tile/tile_mmad.hpp"

namespace Act::Gemm::Block {

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class TensorA,
    class TensorB,
    class TensorC,
    class TensorBias = void,
    class TileCopy = Gemm::Tile::PackedTileCopyTla<typename DispatchPolicy::ArchTag, TensorA, layout::RowMajor,
        TensorB, layout::RowMajor, TensorC, layout::RowMajor, TensorBias, layout::RowMajor>,
    class TileMmad = Gemm::Tile::TileMmadTla<typename DispatchPolicy::ArchTag, typename TileCopy::TensorL0A,
        typename TileCopy::TensorL0B, typename TileCopy::TensorL0C>
>
struct BlockMmadTla {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmadTla is not implemented for this DispatchPolicy");
};

/// new add for the reason that i am using the dispatchpolicy which is same as the policy of the optimized_matmul
// so i add a new one class to avoid the conflict
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopyGemm<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,  // change the name
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockGemm {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace Act::Gemm::Block

#include "act/gemm/block/block_mmad_pingpong.hpp"
#include "act/gemm/block/block_mmad_fa_qk.hpp"
#include "act/gemm/block/block_mmad_fa_pv.hpp"
#include "act/gemm/block/block_mmad_mla_qk.hpp"
#include "act/gemm/block/block_mmad_mla_pv.hpp"
#include "act/gemm/block/block_mmad_mla_qk_tp1_spec.hpp"
#include "act/gemm/block/block_mmad_mla_pv_tp1_spec.hpp"
#include "act/gemm/block/block_mmad_preload.hpp"
#include "act/gemm/block/block_mmad_preload_async.hpp"
#include "act/gemm/block/block_mmad_pingpong_tla.hpp"
#include "act/gemm/block/block_mmad_preload_tla.hpp"
#include "act/gemm/block/block_mmad_preload_async_with_callback.hpp"
#include "act/gemm/block/block_mmad_gemm.hpp"

#endif // ACT_GEMM_BLOCK_BLOCK_MMAD_HPP
