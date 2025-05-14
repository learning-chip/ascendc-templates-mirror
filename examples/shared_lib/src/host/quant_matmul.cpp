#include <iostream> // To find output stream

#include "kernel/quant_matmul.hpp"

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "act_kernel.h"


namespace ActKernel {

    using namespace Act;

    using L1TileShape = GemmShape<128, 256, 512>;
    constexpr uint32_t workspaceStages = 2;

    void QuantMatmul(uint32_t blockNum, aclrtStream stream,KernelInfo kernelInfo) 
    {
        uint32_t m = kernelInfo.m;
        uint32_t n = kernelInfo.n;
        uint32_t k = kernelInfo.k;

        size_t lenWorkspace = static_cast<size_t>(L1TileShape::M) * L1TileShape::N * blockNum * 2;
        size_t sizeWorkspace = lenWorkspace * sizeof(uint32_t);

        uint8_t *deviceWorkspace{nullptr};
        aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), sizeWorkspace, ACL_MEM_MALLOC_HUGE_FIRST);

        GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

        layout::RowMajor layoutA{m, k};
        layout::RowMajor layoutB{k, n};
        layout::VectorLayout layoutScale{n};
        layout::VectorLayout layoutPerTokenScale{m};
        layout::RowMajor layoutC{m, n};

        uint8_t *deviceA = kernelInfo.inputAddr.at(0);
        uint8_t *deviceB = kernelInfo.inputAddr.at(1);
        uint8_t *deviceScale = kernelInfo.inputAddr.at(2);
        uint8_t *devicePerTokenScale = kernelInfo.inputAddr.at(3);

        uint8_t *deviceC = kernelInfo.outputAddr.at(0);

        uint64_t fftsAddr{0};
        uint32_t fftsLen{0};
        rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

        quant_matmul<<<blockNum, nullptr, stream>>>(
        fftsAddr,
        problemShape,
        deviceA, layoutA,
        deviceB, layoutB,
        deviceScale, layoutScale,
        devicePerTokenScale, layoutPerTokenScale,
        deviceC, layoutC,
        deviceWorkspace
        );

        aclrtFree(deviceWorkspace);

        aclrtSynchronizeStream(stream);
    }

}  // namespace ActKernel