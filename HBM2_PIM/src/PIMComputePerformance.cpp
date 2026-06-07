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

#include "PIMComputePerformance.h"
#include "gtest/gtest.h"
#include "tests/PIMCmdGen.h"
#include <fstream>

#if 0

using namespace DRAMSim;
using namespace std;

PIMPureComputePerformanceFixture::PIMPureComputePerformanceFixture()
{
    // 初始化系统
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm_64ch.ini", ".", "pim_pure_compute_perf_test", 256 * 16);

    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
    precision = PIMConfiguration::getPIMPrecision();

    num_grf_ = 8;  // 每个 PIM 块有 8 个 GRF 寄存器
    num_pim_blocks_ = num_chans * getConfigParam(UINT, "NUM_PIM_BLOCKS");
    null_bst_ = BurstType();  // 初始化 null_bst_
}

PIMPureComputePerformanceFixture::~PIMPureComputePerformanceFixture()
{
    // pim_mem_->printStats(true);
}

void PIMPureComputePerformanceFixture::run()
{
    // 使用与 PIMKernel::runPIM() 相同的方式来计数周期
    while (pim_mem_->hasPendingTransactions())
    {
        kernel_->runPIM();  // 这会自动调用 mem_->update() 并递增 cycle_
    }
}

string PIMPureComputePerformanceFixture::getUnit()
{
    return (precision == INT8) ? "GOPS" : "GFLOPS";
}

void PIMPureComputePerformanceFixture::testPureAddPerformance(int num_elements)
{
    cout << ">> PIM Pure Compute Performance Test (ADD)" << endl;
    cout << "> Channels: " << num_chans << endl;
    cout << "> Precision: " << ((precision == FP16) ? "FP16" : (precision == FP32 ? "FP32" : "INT8")) << endl;
    cout << "> Num Elements: " << num_elements << endl;

    // 使用 DataDim 管理维度和数据
    DataDim dim(KernelType::ADD, 1, num_elements, num_elements, false);

    // 补全 Dummy 数据空间，防止模拟器访问空 vector 崩溃
    dim.input_npbst_.bData.assign(dim.input_npbst_.getTotalDim(), BurstType());
    dim.output_npbst_.bData.assign(dim.output_npbst_.getTotalDim(), BurstType());

    // 记录开始前的周期
    uint64_t start_cycle = kernel_->getCycle();

    // 1. 触发 PIM 算子
    kernel_->executeEltwise(dim.output_npbst_.getTotalDim(), pimBankType::ALL_BANK, KernelType::ADD, 0, 100, 200);

    // 2. 驱动执行并计数
    kernel_->runPIM();

    uint64_t end_cycle = kernel_->getCycle();
    uint64_t duration_cycles = end_cycle - start_cycle;
    double time_sec = (double)duration_cycles * tCK * 1e-9;

    // 计算 Ops (ADD 每个元素 1 次运算)
    double total_ops = (double)num_elements;
    double performance = (duration_cycles > 0) ? (total_ops / time_sec / 1e9) : 0;

    cout << "> Results:" << endl;
    cout << "  Cycles: " << duration_cycles << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Performance: " << performance << " " << getUnit() << endl;
    cout << "------------------------------------------------" << endl;
}

void PIMPureComputePerformanceFixture::testPureMulPerformance(int num_elements)
{
    cout << ">> PIM Pure Compute Performance Test (MUL)" << endl;
    cout << "> Channels: " << num_chans << endl;
    cout << "> Precision: " << ((precision == FP16) ? "FP16" : (precision == FP32 ? "FP32" : "INT8")) << endl;
    cout << "> Num Elements: " << num_elements << endl;

    // 使用 DataDim 管理维度和数据
    DataDim dim(KernelType::MUL, 1, num_elements, num_elements, false);

    // 补全 Dummy 数据空间，防止模拟器访问空 vector 崩溃
    dim.input_npbst_.bData.assign(dim.input_npbst_.getTotalDim(), BurstType());
    dim.output_npbst_.bData.assign(dim.output_npbst_.getTotalDim(), BurstType());

    // 记录开始前的周期
    uint64_t start_cycle = kernel_->getCycle();

    // 1. 触发 PIM 算子
    kernel_->executeEltwise(dim.output_npbst_.getTotalDim(), pimBankType::ALL_BANK, KernelType::MUL, 0, 100, 200);

    // 2. 驱动执行并计数
    kernel_->runPIM();

    uint64_t end_cycle = kernel_->getCycle();
    uint64_t duration_cycles = end_cycle - start_cycle;
    double time_sec = (double)duration_cycles * tCK * 1e-9;

    // 计算 Ops (MUL 每个元素 1 次运算)
    double total_ops = (double)num_elements;
    double performance = (duration_cycles > 0) ? (total_ops / time_sec / 1e9) : 0;

    cout << "> Results:" << endl;
    cout << "  Cycles: " << duration_cycles << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Performance: " << performance << " " << getUnit() << endl;
    cout << "------------------------------------------------" << endl;
}

void PIMPureComputePerformanceFixture::testPureMacPerformance(int rows, int cols)
{
    cout << ">> PIM Pure Compute Performance Test (GEMV/MAC)" << endl;
    cout << "> Channels: " << num_chans << endl;
    cout << "> Precision: " << ((precision == FP16) ? "FP16" : (precision == FP32 ? "FP32" : "INT8")) << endl;
    cout << "> Matrix: " << rows << "x" << cols << endl;

    // 使用 DataDim 管理维度和数据
    DataDim dim(KernelType::GEMV, 1, rows, cols, false);

    // 【关键修复】：手动分配权重矩阵和输入向量的 Dummy 数据空间
    dim.weight_npbst_.bData.assign(dim.weight_npbst_.getTotalDim(), BurstType());
    dim.input_npbst_.bData.assign(dim.input_npbst_.getTotalDim(), BurstType());

    // 预热权重
    kernel_->preloadGemv(&dim.weight_npbst_, 0, 0);
    kernel_->runPIM();

    uint64_t start_cycle = kernel_->getCycle();

    // 1. 触发 GEMV 算子
    kernel_->executeGemv(&dim.weight_npbst_, &dim.input_npbst_, false);

    // 2. 驱动执行并计数
    kernel_->runPIM();

    uint64_t end_cycle = kernel_->getCycle();
    uint64_t duration_cycles = end_cycle - start_cycle;
    double time_sec = (double)duration_cycles * tCK * 1e-9;

    // 计算 Ops: 2 * rows * cols (1 MAC = 2 Ops)
    double total_ops = 2.0 * (double)rows * (double)cols;
    double performance = (duration_cycles > 0) ? (total_ops / time_sec / 1e9) : 0;

    cout << "> Results:" << endl;
    cout << "  Cycles: " << duration_cycles << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Performance: " << performance << " " << getUnit() << endl;
    cout << "------------------------------------------------" << endl;
}

void PIMPureComputePerformanceFixture::setupScaling(int num_pcus)
{
    pim_mem_.reset();
    kernel_.reset();

    string device_ini_src = "ini/HBM2_samsung_2M_16B_x64.ini";
    string device_ini_dst = "ini/HBM2_samsung_temp.ini";

    ifstream src(device_ini_src);
    ofstream dst(device_ini_dst);
    string line;
    if (!src.is_open()) {
        cerr << "Failed to open source INI: " << device_ini_src << endl;
        return;
    }
    while (getline(src, line))
    {
        if (line.find("NUM_PIM_BLOCKS") != string::npos)
        {
            dst << "NUM_PIM_BLOCKS=" << num_pcus << endl;
        }
        else
        {
            dst << line << endl;
        }
    }
    dst.close();
    src.close();

    // Re-initialize system with 1 channel and modified device INI
    pim_mem_ = make_shared<MultiChannelMemorySystem>(device_ini_dst, "system_hbm_1ch.ini", ".", "pim_scaling_test", 256 * 16);

    num_chans = getConfigParam(UINT, "NUM_CHANS");
    // Usually 1 rank for HBM
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
    precision = PIMConfiguration::getPIMPrecision();
    num_grf_ = 8;
    num_pim_blocks_ = num_chans * getConfigParam(UINT, "NUM_PIM_BLOCKS");
    
    cout << ">> Setup Scaling: 1 Channel, " << num_pcus << " PCUs (" << num_pim_blocks_ << " Total Blocks)" << endl;
}

// ============================================================================
// Google Test Cases
// ============================================================================

TEST_F(PIMPureComputePerformanceFixture, pure_scaling_1ch)
{
    int pcus[] = {1, 2, 4, 8};
    for (int p : pcus)
    {
        setupScaling(p);
        cout << "\n==============================================" << endl;
        cout << " Testing with " << p << " PCUs in 1 Channel" << endl;
        cout << "==============================================" << endl;
        
        // Use enough workload to saturate the compute
        testPureAddPerformance(1024 * 1024);
        testPureMulPerformance(1024 * 1024);
        testPureMacPerformance(4096, 4096);
    }
}

// 中等元素数量测试（1M 元素）
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
    testPureMacPerformance(2048, 4096);
}

#endif
