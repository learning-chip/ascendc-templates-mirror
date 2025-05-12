#ifndef SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H
#define SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "act_kernel.h"

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/epilogue/block/block_epilogue.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/epilogue/tile/tile_broadcast_mul.hpp"
#include "act/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "act/epilogue/tile/tile_swizzle.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "act/gemm/gemm_type.hpp"
#include "act/layout/layout.hpp"

#include "../types/bfloat16.h"

using namespace Act;

template <
    class LayoutA,
    class LayoutB
>
ACT_GLOBAL
void quant_matmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmScale, layout::VectorLayout layoutScale,
    GM_ADDR gmPerTokenScale, layout::VectorLayout layoutPerTokenScale,
    GM_ADDR gmD, layout::RowMajor layoutD,
    GM_ADDR gmWorkspace
)
{
  using bfloat16 = op::bfloat16;

  using L1TileShape = GemmShape<128, 256, 512>;
  constexpr uint32_t workspaceStages = 2;

    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Arch::AtlasA2;
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L0TileShape = GemmShape<128, 256, 128>;

    using AType = Gemm::GemmType<int8_t, LayoutA>;
    using BType = Gemm::GemmType<int8_t, LayoutB>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<bfloat16_t, layout::VectorLayout>;
    using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = MatrixShape<32, 256>;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using BlockScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, BlockScheduler>;

    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;

        typename MatmulKernel::Params params{
            problemShape,
            gmA, layoutA,
            gmB, layoutB,
            gmScale, layoutScale,
            gmPerTokenScale, layoutPerTokenScale,
            gmD, layoutD,
            gmWorkspace
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::QuantMatmulMultiStageWorkspace<BlockMmad, BlockEpilogue,
            BlockScheduler, workspaceStages>;

        typename MatmulKernel::Params params{
            problemShape,
            gmA, layoutA,
            gmB, layoutB,
            gmScale, layoutScale,
            gmPerTokenScale, layoutPerTokenScale,
            gmD, layoutD,
            gmWorkspace
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
} 

#endif // SHARED_LIB_IMPL_OPTIMIZED_MATMUL_H