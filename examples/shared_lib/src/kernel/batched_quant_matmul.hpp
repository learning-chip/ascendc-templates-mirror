
#ifndef SHARED_LIB_IMPL_BATCHED_QUANT_MATMUL_H
#define SHARED_LIB_IMPL_BATCHED_QUANT_MATMUL_H

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/batched_matmul.hpp"
#include "act/gemm/gemm_type.hpp"
#include "act/layout/layout.hpp"

using namespace Act;

template <class LayoutA, class LayoutB, class LayoutC>
ACT_GLOBAL
void batched_quant_matmul(uint32_t batchCount, GemmCoord problemShape,
                   GM_ADDR gmA, LayoutA layoutA,
                   GM_ADDR gmB, LayoutB layoutB,
                   GM_ADDR gmC, LayoutC layoutC, float quantScale)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = GemmShape<128, 256, 512>;
    using L0TileShape = GemmShape<128, 256, 128>;

    using AType = Gemm::GemmType<int8_t, LayoutA>;
    using BType = Gemm::GemmType<int8_t, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    if (problemShape.m() > problemShape.n()) {
        // Swizzle offset is 3 and direction is 0.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        typename MatmulKernel::Params params{
            batchCount, problemShape,
            gmA, layoutA, strideA,
            gmB, layoutB, strideB,
            gmC, layoutC, strideC, quantScale
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    } else {
        // Swizzle offset is 3 and direction is 1.
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::BatchedMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

        int64_t strideA = problemShape.m() * problemShape.k();
        int64_t strideB = problemShape.k() * problemShape.n();
        int64_t strideC = problemShape.m() * problemShape.n();
        typename MatmulKernel::Params params{
            batchCount, problemShape,
            gmA, layoutA, strideA,
            gmB, layoutB, strideB,
            gmC, layoutC, strideC, quantScale
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

#endif // SHARED_LIB_IMPL_BATCHED_QUANT_MATMUL_H