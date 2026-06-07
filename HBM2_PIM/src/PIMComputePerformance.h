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

#include <memory>
#include <string>
#include <vector>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMPureComputePerformanceFixture: 极限计算性能测试
 *
 * 目标：
 * - 测试 PIM 块的纯计算性能（消除内存访问开销）
 * - 数据已经在 GRF（全局寄存器文件）中，连续执行计算操作
 * - 不包含 BANK_TO_GRF 和 GRF_TO_BANK 数据搬运阶段
 * - 测量真正的计算极限性能
 *
 * 测试方法：
 * - 初始化时将数据加载到 GRF 中（一次性内存访问）
 * - 连续执行大量计算操作（ADD/MUL/MAC）
 * - 不将结果写回内存
 * - 计算纯计算阶段的性能
 *
 * 关键优化：
 * - 消除内存带宽瓶颈
 * - 充分利用 PIM 块的并行性
 * - 测量持续计算的稳定性能
 */

namespace DRAMSim
{
class PIMPureComputePerformanceFixture : public testing::Test
{
  public:
    PIMPureComputePerformanceFixture();
    virtual ~PIMPureComputePerformanceFixture();

    // 测量纯 ADD 性能
    void testPureAddPerformance(int num_elements);

    // 测量纯 MUL 性能
    void testPureMulPerformance(int num_elements);

    // 测量纯 MAC 性能（GEMV）
    void testPureMacPerformance(int rows, int cols);

    // Scaling Setup
    void setupScaling(int num_pcus);

  protected:
    void run();
    std::string getUnit();
    void initializeGRF();  // 初始化 GRF 数据（一次性内存访问）

    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
    PIMPrecision precision;
    unsigned num_grf_;  // 每个 PIM 块的 GRF 大小
    unsigned num_pim_blocks_;  // 总 PIM 块数量
    BurstType null_bst_;  // 用于无数据传输的 null 突发
};

}  // namespace DRAMSim

#endif /*__PIM_PURE_COMPUTE_PERFORMANCE_TEST_CASE_H__*/
