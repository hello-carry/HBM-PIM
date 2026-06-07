/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef __PIM_PURE_COMPUTE_PERFORMANCE_TEST_CASE_H__
#define __PIM_PURE_COMPUTE_PERFORMANCE_TEST_CASE_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMPureComputePerformanceFixture: peak compute performance test
 *
 * Goals:
 * - Test the ALU-only behavior of HBM2-PIM FP16 compute throughput
 * - Include only CRF arithmetic triggers under HAB_PIM in the measurement window
 * - Keep SB/HAB/HAB_PIM mode switches, CRF programming, and Bank/GRF transfers
 *   outside the measurement window
 *
 * Test method:
 * - Use a benchmark-local FP16 config to avoid being affected by the FP32/INT8
 *   settings in system_hbm.ini
 * - Calculate actual ops using lanes-per-burst for the current precision
 * - For GEMV/MAC, align the actual executed MAC ops with PIM block/lane
 *   parallelism
 *
 * Key optimizations:
 * - Remove the memory-bandwidth bottleneck
 * - Fully utilize PIM block parallelism
 * - Measure stable performance for sustained computation
 */

namespace DRAMSim
{
class PIMPureComputePerformanceFixture : public testing::Test
{
  public:
    PIMPureComputePerformanceFixture();
    virtual ~PIMPureComputePerformanceFixture();

    // Measure pure ADD performance
    void testPureAddPerformance(int num_elements);

    // Measure pure MUL performance
    void testPureMulPerformance(int num_elements);

    // Measure pure MAC performance (GEMV)
    void testPureMacPerformance(int rows, int cols);

  protected:
    void run();
    std::string getUnit() const;
    std::string getPrecisionName() const;
    unsigned getLanesPerBurst() const;
    uint64_t getTotalPIMBlocks() const;
    uint64_t getOpsPerInstruction(KernelType ktype) const;
    uint64_t getBurstCount(const NumpyBurstType& tensor) const;
    uint64_t getElementwiseOps(const NumpyBurstType& output) const;
    uint64_t getGemvOps(const NumpyBurstType& weight) const;
    void runPureCompute(KernelType ktype, uint64_t requested_ops, const std::string& op_name);
    void printExecutionScope(const std::string& path_name, uint64_t ops_per_instruction,
                             uint64_t iterations) const;
    void initializeGRF();  // Initialize GRF data (one-time memory access)

    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
    PIMPrecision precision;
    unsigned num_grf_;  // GRF size per PIM block
    unsigned num_pim_blocks_;  // Total number of PIM blocks
    BurstType null_bst_;  // Null burst used when no data is transferred
};

}  // namespace DRAMSim

#endif /*__PIM_PURE_COMPUTE_PERFORMANCE_TEST_CASE_H__*/
