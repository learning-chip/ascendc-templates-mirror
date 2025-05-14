/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0, 
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <iostream>
#include <vector>
#include <random>

#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"

#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/layout/layout.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/optimized_matmul.hpp"
#include "act/gemm/gemm_type.hpp"

using namespace Act;
using fp16_t = op::fp16_t;
using InputType = int8_t;
using OutputTypeKernel = half;
using OutputTypeCPP = fp16_t;

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
void OptimizedQuantMatmul(
    uint64_t fftsAddr,
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC,
    GM_ADDR gmWA, GM_ADDR gmWB, float quantScale
)
{
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

struct Options {
    const std::string HELPER = "21_optimized_quant_matmul m n k [device_id]";

    GemmCoord problemShape{128, 128, 128};
    int32_t deviceId{0};

    Options() = default;

    int Parse(int argc, const char **argv)
    {
        enum ArgsIndex {
            M_INDEX = 1,
            N_INDEX,
            K_INDEX,
            DEVICE_ID_INDEX,
            ARGS_MAX
        };

        if (argc > ARGS_MAX || argc <= K_INDEX) {
            std::cerr << HELPER << std::endl;
            return -1;
        }

        problemShape.m() = std::atoi(argv[M_INDEX]);
        problemShape.n() = std::atoi(argv[N_INDEX]);
        problemShape.k() = std::atoi(argv[K_INDEX]);
        if (argc == ARGS_MAX) {
            deviceId = std::atoi(argv[DEVICE_ID_INDEX]);
        }
        return 0;
    }
};

template<class Layout>
size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
{
    return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
        RoundUp(static_cast<size_t>(layout.shape(1)), blockCols);
}

bool IsNeedPadding(layout::RowMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(0) < 65536) {
        return layout.stride(0) % align != 0;
    } else {
        return true;
    }
}

bool IsNeedPadding(layout::ColumnMajor layout, uint32_t align)
{
    // If the stride is greater than 65536, padding is required to reduce the stride.
    if (layout.stride(1) < 65536) {
        return layout.stride(1) % align != 0;
    } else {
        return true;
    }
}

void Run(Options const &options)
{
    aclrtStream stream{nullptr};

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    uint32_t m = options.problemShape.m();
    uint32_t n = options.problemShape.n();
    uint32_t k = options.problemShape.k();

    size_t lenA = static_cast<size_t>(m) * k;
    size_t lenB = static_cast<size_t>(k) * n;
    size_t lenC = static_cast<size_t>(m) * n;

    size_t sizeA = lenA * sizeof(InputType);
    size_t sizeB = lenB * sizeof(InputType);
    size_t sizeC = lenC * sizeof(OutputTypeCPP);

    const uint32_t align = 256;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
    size_t sizeWA = GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(InputType);
    size_t sizeWB = GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(InputType);

    std::vector<InputType> hostA(lenA);
    std::vector<InputType> hostB(lenB);
    golden::FillRandomData<InputType>(hostA, -5.0, 5.0);
    golden::FillRandomData<InputType>(hostB, -5.0, 5.0);

    uint8_t *deviceA{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceB{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceC{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST));
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-16.0f, 16.0f);

    float quantScale = dist(gen);

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    OptimizedQuantMatmul<<<aicCoreNum, nullptr, stream>>>(
        fftsAddr,
        options.problemShape, deviceA, layoutA, deviceB, layoutB, deviceC, layoutC,
        deviceWA, deviceWB, quantScale);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<OutputTypeCPP> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

    std::vector<float> hostGolden(lenC);
    golden::ComputeMatmul(options.problemShape, hostA, layoutA, hostB, layoutB, hostGolden, layoutC);

    // Multiply the golden result by quantScale
    for (size_t i = 0; i < lenC; ++i) {
        hostGolden[i] *= quantScale;
    }

    std::vector<uint64_t> errorIndices = golden::CompareData(hostC, hostGolden, k);
    if (errorIndices.empty()) {
        std::cout << "Compare success." << std::endl;
    } else {
        std::cerr << "Compare failed. Error count: " << errorIndices.size() << std::endl;
    }

    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    if (isNeedPaddingA) {
        ACL_CHECK(aclrtFree(deviceWA));
    }
    if (isNeedPaddingB) {
        ACL_CHECK(aclrtFree(deviceWB));
    }
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

int main(int argc, const char **argv)
{
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}
