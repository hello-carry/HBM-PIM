/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#include "PIMReduction.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace DRAMSim;
using namespace std;

PIMReductionBenchFixture::PIMReductionBenchFixture()
{
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_reduction", 256 * 16);
    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
}

PIMReductionBenchFixture::~PIMReductionBenchFixture() {}

void PIMReductionBenchFixture::run() { while (pim_mem_->hasPendingTransactions()) pim_mem_->update(); }

void PIMReductionBenchFixture::testSumReduction(int total_elements)
{
    cout << ">> PIM Sum Reduction Bench (tree, stride-doubling per pass)" << endl;
    cout << "> Channels: " << num_chans << " | Total Elements: " << total_elements << endl;

    const unsigned num_grf            = (unsigned)kernel_->num_grf_;                       // 8
    const unsigned num_banks          = getConfigParam(UINT, "NUM_BANKS");                 // 16
    const unsigned num_banks_per_side = num_banks / 2;                                     // 8 (even or odd)
    const unsigned lanes_per_burst    = 16;                                                // fp16 lanes per burst
    const unsigned num_cols_per_bl    = kernel_->pim_addr_mgr_->num_cols_per_bl_;          // NUM_COLS/BL = 32
    const unsigned num_rows           = kernel_->pim_addr_mgr_->num_rows_;                 // 16384
    const uint64_t bank_burst_cap     = (uint64_t)num_cols_per_bl * num_rows;              // 524288

    // Each shot = one full CRF round (FILL+ADD) x (EVEN+ODD), full-channel x full-bank fan-out
    const uint64_t pairs_per_shot =
        (uint64_t)num_chans * num_banks_per_side * num_grf * lanes_per_burst * 2;

    // CRF: even+odd accumulate (following EltwisePIMKernel::generateKernel) + JUMP loop
    vector<PIMCmd> cmds = {
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK),
        PIMCmd(PIMCmdType::ADD,  PIMOpdType::GRF_A, PIMOpdType::GRF_A, PIMOpdType::EVEN_BANK, 1),
        PIMCmd(PIMCmdType::NOP,  7),
        PIMCmd(PIMCmdType::FILL, PIMOpdType::GRF_B, PIMOpdType::ODD_BANK),
        PIMCmd(PIMCmdType::ADD,  PIMOpdType::GRF_B, PIMOpdType::GRF_B, PIMOpdType::ODD_BANK, 1),
        PIMCmd(PIMCmdType::NOP,  7),
    };
    // Keep num_jump large enough; the host controls when triggers stop, and each pass resets PC through HAB->HAB_PIM switching
    const int kMaxJump = 65535;
    cmds.push_back(PIMCmd(PIMCmdType::JUMP, kMaxJump, (int)cmds.size() + 1));
    cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));

    kernel_->setPIMControl(/*pim_op=*/true, /*crf_toggle_cond=*/0 /*ALL_BANK*/, false, false);
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(cmds);

    uint64_t total_cycles = 0;
    BurstType trigger;
    int step = 0;
    while (((uint64_t)total_elements >> (step + 1)) > 0)
    {
        const uint64_t stride_bursts = (uint64_t)1 << step;
        const uint64_t pairs_total   = (uint64_t)total_elements >> (step + 1);
        uint64_t shots = (pairs_total + pairs_per_shot - 1) / pairs_per_shot;
        if (shots == 0) shots = 1;

        // PC reset: HAB_PIM<->HAB toggle makes CRF restart from PC=0
        kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);

        uint64_t step_start = kernel_->getCycle();
        cout << "  - Step " << step << ": Stride=" << stride_bursts
             << " burst | Pairs=" << pairs_total << " | Shots=" << shots << endl;

        for (uint64_t i = 0; i < shots; i++)
        {
            // Shot i handles num_grf burst pairs; left and right in each pair are stride_bursts apart
            // Wrap within the physical bank range for large strides to avoid row overflow
            const uint64_t left_burst  = ((uint64_t)i * num_grf) % bank_burst_cap;
            const uint64_t right_burst = (left_burst + stride_bursts) % bank_burst_cap;

            const unsigned col_left  = (unsigned)(left_burst  % num_cols_per_bl);
            const unsigned row_left  = (unsigned)((left_burst  / num_cols_per_bl) % num_rows);
            const unsigned col_right = (unsigned)(right_burst % num_cols_per_bl);
            const unsigned row_right = (unsigned)((right_burst / num_cols_per_bl) % num_rows);

            // EVEN bank: 8 trans FILL (PC0/1) + 8 trans ADD (PC1/2)
            kernel_->addTransactionAll(false, 0, 0, row_left,  col_left,  &trigger, false, num_grf);
            kernel_->addTransactionAll(false, 0, 0, row_right, col_right, &trigger, false, num_grf);
            // ODD bank: 8 trans FILL (PC3/4) + 8 trans ADD (PC4/5)
            kernel_->addTransactionAll(false, 0, 1, row_left,  col_left,  &trigger, false, num_grf);
            kernel_->addTransactionAll(false, 0, 1, row_right, col_right, &trigger, false, num_grf);
        }

        kernel_->runPIM();
        total_cycles += (kernel_->getCycle() - step_start);

        kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
        step++;
    }

    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    kernel_->parkOut();

    double time_ms = (double)total_cycles * tCK * 1e-6;
    cout << "> Results:" << endl;
    cout << "  Total Cycles: " << total_cycles << endl;
    cout << "  Simulated Time: " << time_ms << " ms" << endl;
    cout << "  Steps Done: " << step << endl;
    cout << "------------------------------------------------" << endl;
}

// ============================================================================
// Test Cases
// ============================================================================

TEST_F(PIMReductionBenchFixture, reduction_1M) {
    // testMaxReduction(6606029);
    testSumReduction(1000 * 1000);
}

TEST_F(PIMReductionBenchFixture, reduction_10M) {
    // testMaxReduction(6606029);
    testSumReduction(10 * 1000 * 1000);
}

TEST_F(PIMReductionBenchFixture, reduction_63M) {
    // testMaxReduction(6606029);
    testSumReduction(63 * 1000 * 1000);
}

// TEST_F(PIMReductionBenchFixture, reduction_400M) {
//     testSumReduction(400 * 1024 * 1024);
// }