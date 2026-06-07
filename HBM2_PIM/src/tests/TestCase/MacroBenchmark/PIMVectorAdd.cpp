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

#include "tests/TestCase/MacroBenchmark/PIMVectorAdd.h"
#include "gtest/gtest.h"

using namespace DRAMSim;
using namespace std;

PIMVectorAddFixture::PIMVectorAddFixture()
{
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_vadd_macro", 256 * 16);
    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
    precision = PIMConfiguration::getPIMPrecision();
}

PIMVectorAddFixture::~PIMVectorAddFixture() {}

void PIMVectorAddFixture::run()
{
    while (pim_mem_->hasPendingTransactions()) {
        pim_mem_->update();
    }
}

void PIMVectorAddFixture::testVectorAdd(uint64_t num_elements)
{
    cout << ">> MacroBench: Vector Add (A + B = C)" << endl;
    cout << "> Channels: " << num_chans << " | Elements: " << num_elements << endl;

    DataDim dim(KernelType::ADD, 1, num_elements, num_elements, false);
    dim.input_npbst_.bData.assign(dim.input_npbst_.getTotalDim(), BurstType());
    dim.output_npbst_.bData.assign(dim.output_npbst_.getTotalDim(), BurstType());

    uint64_t start_cycle = kernel_->getCycle();
    kernel_->executeEltwise(dim.output_npbst_.getTotalDim(),
                            pimBankType::ALL_BANK, KernelType::ADD,
                            /*input0_row=*/0, /*result_row=*/200, /*input1_row=*/100);
    kernel_->runPIM();
    uint64_t pim_cycles = kernel_->getCycle() - start_cycle;

    cout << "  [PIM Enabled ] Cycles: " << pim_cycles << " (" << (pim_cycles * tCK * 1e-6) << " ms)" << endl;
    cout << "------------------------------------------------" << endl;
}

TEST_F(PIMVectorAddFixture, vadd_1M) { testVectorAdd(1 * 1024 * 1024); }

TEST_F(PIMVectorAddFixture, vadd_10M) { testVectorAdd(10 * 1024 * 1024); }

TEST_F(PIMVectorAddFixture, vadd_16M) { testVectorAdd(16 * 1024 * 1024); }

