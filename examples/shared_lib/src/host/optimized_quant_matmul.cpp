#include "kernel/optimized_quant_matmul.hpp"

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "act_kernel.h"

namespace ActKernel {
using namespace Act;

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

void OptimizedQuantMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;
    float quantizationScale = kernelInfo.quantizationScale;

    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

    const uint32_t align = 128;
    using LayoutA = layout::RowMajor;
    using LayoutB = layout::ColumnMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    bool isNeedPaddingA = IsNeedPadding(layoutA, align);
    bool isNeedPaddingB = IsNeedPadding(layoutB, align);

    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 512>, GemmShape<128, 256, 512>>;
    size_t sizeWA = GetWorkspaceLen(layoutA, L1TileShape::M, L1TileShape::K) * sizeof(int8_t);
    size_t sizeWB = GetWorkspaceLen(layoutB, L1TileShape::K, L1TileShape::N) * sizeof(int8_t);


    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    uint8_t *deviceWA{nullptr};
    if (isNeedPaddingA) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWA), sizeWA, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding A
        deviceWA = deviceA;
    }

    uint8_t *deviceWB{nullptr};
    // If layoutWB has the same stride with layoutB, no need to padding B
    if (isNeedPaddingB) {
        aclrtMalloc(reinterpret_cast<void **>(&deviceWB), sizeWB, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        // no need to padding B
        deviceWB = deviceB;
    }

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    optimized_quant_matmul<<<blockNum, nullptr, stream>>>(
        fftsAddr,
        problemShape, 
        deviceA, layoutA, 
        deviceB, layoutB, 
        deviceC, layoutC,
        deviceWA, deviceWB, 
        quantizationScale);

    if (isNeedPaddingA) {
        aclrtFree(deviceWA);
    }
    if (isNeedPaddingB) {
        aclrtFree(deviceWB);
    }
}

}  // namespace ActKernel