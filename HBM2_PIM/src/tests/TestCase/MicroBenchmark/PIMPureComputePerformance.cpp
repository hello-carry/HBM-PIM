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

#include "PIMPureComputePerformance.h"
#include "gtest/gtest.h"
#include "tests/PIMCmdGen.h"

using namespace DRAMSim;
using namespace std;

namespace
{
const char* kPureComputeSystemIni = "system_hbm_pim_fp16.ini";

uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}
}

PIMPureComputePerformanceFixture::PIMPureComputePerformanceFixture()
{
    // Use a benchmark-local FP16 config. HBM2-PIM peak compute is normally reported
    // in FP16 lanes; inheriting system_hbm.ini can silently switch this test to FP32.
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     kPureComputeSystemIni, ".",
                                                     "pim_pure_compute_perf_test", 256 * 16);

    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
    precision = PIMConfiguration::getPIMPrecision();

    num_grf_ = 8;  // Each PIM block has 8 GRF registers
    num_pim_blocks_ = num_chans * getConfigParam(UINT, "NUM_PIM_BLOCKS");
    null_bst_ = BurstType();  // Initialize null_bst_
}

PIMPureComputePerformanceFixture::~PIMPureComputePerformanceFixture()
{
    // pim_mem_->printStats(true);
}

void PIMPureComputePerformanceFixture::run()
{
    // Count cycles the same way as PIMKernel::runPIM()
    while (pim_mem_->hasPendingTransactions())
    {
        kernel_->runPIM();  // This automatically calls mem_->update() and increments cycle_
    }
}

string PIMPureComputePerformanceFixture::getUnit() const
{
    return (precision == INT8) ? "GOPS" : "GFLOPS";
}

string PIMPureComputePerformanceFixture::getPrecisionName() const
{
    switch (precision)
    {
        case FP16:
            return "FP16";
        case FP32:
            return "FP32";
        case INT8:
            return "INT8";
        default:
            return "UNKNOWN";
    }
}

unsigned PIMPureComputePerformanceFixture::getLanesPerBurst() const
{
    const unsigned bytes_per_burst =
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8;
    return bytes_per_burst / PIMConfiguration::getPIMDataLength();
}

uint64_t PIMPureComputePerformanceFixture::getTotalPIMBlocks() const
{
    return static_cast<uint64_t>(num_chans) * getConfigParam(UINT, "NUM_PIM_BLOCKS");
}

uint64_t PIMPureComputePerformanceFixture::getOpsPerInstruction(KernelType ktype) const
{
    const uint64_t op_factor = (ktype == KernelType::GEMV) ? 2ULL : 1ULL;
    return getTotalPIMBlocks() * getLanesPerBurst() * op_factor;
}

uint64_t PIMPureComputePerformanceFixture::getBurstCount(const NumpyBurstType& tensor) const
{
    uint64_t bursts = 1;
    for (unsigned long dim : tensor.bShape) bursts *= dim;
    return bursts;
}

uint64_t PIMPureComputePerformanceFixture::getElementwiseOps(const NumpyBurstType& output) const
{
    return getBurstCount(output) * getLanesPerBurst();
}

uint64_t PIMPureComputePerformanceFixture::getGemvOps(const NumpyBurstType& weight) const
{
    if (weight.bShape.size() < 2) return 0;
    const uint64_t output_rows = weight.bShape[0];
    const uint64_t input_bursts_per_row = weight.bShape[1];
    return 2ULL * output_rows * input_bursts_per_row * getLanesPerBurst();
}

void PIMPureComputePerformanceFixture::printExecutionScope(const string& path_name,
                                                          uint64_t ops_per_instruction,
                                                          uint64_t iterations) const
{
    cout << "> Benchmark Config: " << kPureComputeSystemIni << endl;
    cout << "> Channels: " << num_chans << endl;
    cout << "> PIM Blocks: " << getTotalPIMBlocks()
         << " total (" << getConfigParam(UINT, "NUM_PIM_BLOCKS") << "/channel)" << endl;
    cout << "> Precision: " << getPrecisionName()
         << " (" << getLanesPerBurst() << " lanes/burst)" << endl;
    cout << "> Scope: " << path_name << " ALU-only trigger window" << endl;
    cout << "> Ops/PIM Instruction: " << ops_per_instruction << endl;
    cout << "> PIM Instruction Iterations: " << iterations << endl;
}

void PIMPureComputePerformanceFixture::runPureCompute(KernelType ktype, uint64_t requested_ops,
                                                      const string& op_name)
{
    const uint64_t ops_per_instruction = getOpsPerInstruction(ktype);
    const uint64_t iterations = ceilDiv(requested_ops, ops_per_instruction);
    const uint64_t total_ops = iterations * ops_per_instruction;
    PIMCmdType cmd_type = PIMCmdType::ADD;

    if (ktype == KernelType::ADD)
        cmd_type = PIMCmdType::ADD;
    else if (ktype == KernelType::MUL)
        cmd_type = PIMCmdType::MUL;
    else if (ktype == KernelType::GEMV)
        cmd_type = PIMCmdType::MAC;
    else
        throw invalid_argument("Unsupported pure compute kernel type");

    printExecutionScope("CRF " + op_name + " GRF-GRF", ops_per_instruction, iterations);

    vector<PIMCmd> cmds;
    if (cmd_type == PIMCmdType::MAC)
    {
        cmds.push_back(PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                              PIMOpdType::GRF_B, 0));
    }
    else
    {
        cmds.push_back(PIMCmd(cmd_type, PIMOpdType::GRF_A, PIMOpdType::GRF_A,
                              PIMOpdType::GRF_B, 0));
    }
    cmds.push_back(PIMCmd(PIMCmdType::JUMP, static_cast<int>(iterations - 1), 2));
    cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));

    BurstType trigger;
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(cmds);
    kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
    kernel_->runPIM();

    uint64_t start_cycle = kernel_->getCycle();
    for (uint64_t i = 0; i < iterations; i++)
    {
        kernel_->addTransactionAll(false, 0, 0, 0, 0, "PURE_" + op_name, &trigger, false, 1);
    }
    kernel_->runPIM();

    uint64_t end_cycle = kernel_->getCycle();
    uint64_t duration_cycles = end_cycle - start_cycle;
    double time_sec = (double)duration_cycles * tCK * 1e-9;
    double performance = (duration_cycles > 0) ? (total_ops / time_sec / 1e9) : 0;

    cout << "> Results:" << endl;
    cout << "  Requested Ops: " << requested_ops << endl;
    cout << "  Effective Ops: " << total_ops << endl;
    cout << "  Trigger Rounds: " << iterations << endl;
    cout << "  Cycles: " << duration_cycles << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Performance: " << performance << " " << getUnit() << endl;
    cout << "------------------------------------------------" << endl;

    kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    kernel_->parkOut();
    kernel_->runPIM();
}

void PIMPureComputePerformanceFixture::testPureAddPerformance(int num_elements)
{
    cout << ">> PIM Pure Compute Performance Test (ADD)" << endl;
    cout << "> Requested Elements: " << num_elements << endl;
    runPureCompute(KernelType::ADD, static_cast<uint64_t>(num_elements), "ADD");
}

void PIMPureComputePerformanceFixture::testPureMulPerformance(int num_elements)
{
    cout << ">> PIM Pure Compute Performance Test (MUL)" << endl;
    cout << "> Requested Elements: " << num_elements << endl;
    runPureCompute(KernelType::MUL, static_cast<uint64_t>(num_elements), "MUL");
}

void PIMPureComputePerformanceFixture::testPureMacPerformance(int rows, int cols)
{
    cout << ">> PIM Pure Compute Performance Test (GEMV/MAC)" << endl;
    cout << "> Matrix: " << rows << "x" << cols << endl;
    const uint64_t requested_ops = 2ULL * rows * cols;
    runPureCompute(KernelType::GEMV, requested_ops, "MAC");
}

// ============================================================================
// Google Test Cases
// ============================================================================

// Medium element-count test (1M elements)
TEST_F(PIMPureComputePerformanceFixture, pure_add_1M)
{
    testPureAddPerformance(1024 * 1024);
}

TEST_F(PIMPureComputePerformanceFixture, pure_mul_1M)
{
    testPureMulPerformance(1024 * 1024);
}

TEST_F(PIMPureComputePerformanceFixture, pure_mac_4K)
{
    testPureMacPerformance(4096, 4096);
}

// // Large element-count test (4M elements)
// TEST_F(PIMPureComputePerformanceFixture, pure_add_4M)
// {
//     testPureAddPerformance(4 * 1024 * 1024);
// }

// TEST_F(PIMPureComputePerformanceFixture, pure_mul_4M)
// {
//     testPureMulPerformance(4 * 1024 * 1024);
// }

// TEST_F(PIMPureComputePerformanceFixture, pure_mac_8K)
// {
//     testPureMacPerformance(8192, 8192);
// }
