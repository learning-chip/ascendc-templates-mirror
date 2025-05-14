#include "kernel/batched_quant_matmul.hpp"

#include <acl/acl.h>
#include <runtime/rt_ffts.h>

#include "act_kernel.h"

namespace ActKernel {
using namespace Act;

void BatchedQuantMatmul(uint32_t blockNum, aclrtStream stream, KernelInfo kernelInfo)
{
    uint32_t m = kernelInfo.m;
    uint32_t n = kernelInfo.n;
    uint32_t k = kernelInfo.k;
    uint32_t batchCount = kernelInfo.batchCount;
    float quantizationScale = kernelInfo.quantizationScale;

    GemmCoord problemShape{kernelInfo.m, kernelInfo.n, kernelInfo.k};

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    uint8_t *deviceA = kernelInfo.inputAddr.at(0);
    uint8_t *deviceB = kernelInfo.inputAddr.at(1);
    uint8_t *deviceC = kernelInfo.outputAddr.at(0);

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    batched_quant_matmul<<<blockNum, nullptr, stream>>>(batchCount, problemShape, 
                                                        deviceA, layoutA, 
                                                        deviceB, layoutB, 
                                                        deviceC, layoutC, 
                                                        quantizationScale);

    // aclrtSynchronizeStream(stream);
}

}  // namespace ActKernel