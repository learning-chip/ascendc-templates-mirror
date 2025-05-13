
#ifndef SHARED_LIB_IMPL_OPTIMIZED_QUANT_MATMUL_H
#define SHARED_LIB_IMPL_OPTIMIZED_QUANT_MATMUL_H

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/optimized_matmul.hpp"
#include "act/gemm/gemm_type.hpp"

using namespace Act;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class LayoutWA,
    class LayoutWB,
    class BlockMmad
>
ACT_DEVICE
void LaunchMatmulDynamicSwizzle(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, LayoutWA layoutWA,
    GM_ADDR gmWB, LayoutWB layoutWB, float quantScale
)
{
    if (problemShape.m() > problemShape.n()) {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB, quantScale};
        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
        using BlockEpilogue = void;
        // kernel level
        using MatmulKernel = Gemm::Kernel::OptimizedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;
        typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC,
            gmWA, layoutWA, gmWB, layoutWB, quantScale};

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
ACT_GLOBAL
void optimized_quant_matmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, GM_ADDR gmWB, float quantScale
)
{
    using InputType = int8_t;
    using OutputTypeKernel = half;
    
    using ArchTag = Arch::AtlasA2;
    AscendC::SetSyncBaseAddr(fftsAddr);

    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2Preload<enableUnitFlag, enableShuffleK>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 512>, GemmShape<128, 256, 512>>;
    using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 128>, GemmShape<128, 256, 128>>;;

    if (gmA == gmWA && gmB == gmWB) {
        // no need to padding A and B.
        using LayoutWA = LayoutA;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<InputType, LayoutWA>;
        using BType = Gemm::GemmType<InputType, LayoutWB>;
        using CType = Gemm::GemmType<OutputTypeKernel, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB, quantScale);
    } else if (gmA == gmWA && gmB != gmWB) {
        // no need to padding A, but B needs padding.
        using LayoutWA = LayoutA;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<InputType, LayoutWA>;
        using BType = Gemm::GemmType<InputType, LayoutWB>;
        using CType = Gemm::GemmType<OutputTypeKernel, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1));
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB, quantScale);
    } else if (gmA != gmWA && gmB == gmWB) {
        // no need to padding B, but A needs padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using LayoutWB = LayoutB;
        using AType = Gemm::GemmType<InputType, LayoutWA>;
        using BType = Gemm::GemmType<InputType, LayoutWB>;
        using CType = Gemm::GemmType<OutputTypeKernel, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1));
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB, quantScale);
    } else {
        // Both A and B need padding.
        using LayoutWA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using LayoutWB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using AType = Gemm::GemmType<InputType, LayoutWA>;
        using BType = Gemm::GemmType<InputType, LayoutWB>;
        using CType = Gemm::GemmType<OutputTypeKernel, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        LayoutWA layoutWA = LayoutWA(layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
        LayoutWB layoutWB = LayoutWB(layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
        LaunchMatmulDynamicSwizzle<LayoutA, LayoutB, LayoutC, LayoutWA, LayoutWB, BlockMmad>(problemShape,
            gmA, layoutA, gmB, layoutB, gmC, layoutC, gmWA, layoutWA, gmWB, layoutWB, quantScale);
    }
}

#endif // SHARED_LIB_IMPL_OPTIMIZED_QUANT_MATMUL_H