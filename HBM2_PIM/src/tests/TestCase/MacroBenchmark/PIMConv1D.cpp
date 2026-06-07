/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#include "PIMConv1D.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "PIMCmd.h"
#include "SystemConfiguration.h"
#include "gtest/gtest.h"

using namespace DRAMSim;
using namespace std;

namespace
{
constexpr int kKernelSize = 10;
constexpr int kStride = 2;
constexpr int kPadding = 4;
constexpr int kMacGroup = 8;
constexpr int kDataRowStride = 4;
constexpr int kResultRowBase = 4096;

int activeLanes()
{
    return (PIMConfiguration::getPIMPrecision() == FP32) ? 8 : 16;
}

float inputValue(int idx)
{
    return 0.125f + static_cast<float>((idx * 13) % 29) / 64.0f;
}

float kernelValue(int idx)
{
    return 0.03125f + static_cast<float>((idx * 7) % 11) / 128.0f;
}

void setLane(BurstType& burst, int lane, float value)
{
    if (PIMConfiguration::getPIMPrecision() == FP32)
        burst.fp32Data_[lane] = value;
    else
        burst.fp16Data_[lane] = convertF2H(value);
}

float getLane(const BurstType& burst, int lane)
{
    if (PIMConfiguration::getPIMPrecision() == FP32)
        return burst.fp32Data_[lane];
    return convertH2F(burst.fp16Data_[lane]);
}

BurstType splatBurst(float value)
{
    BurstType burst;
    const int lanes = activeLanes();
    for (int lane = 0; lane < lanes; lane++) setLane(burst, lane, value);
    return burst;
}

float referenceConvAt(int out_idx, int length)
{
    float acc = 0.0f;
    const int window_base = out_idx * kStride - kPadding;
    for (int k = 0; k < kKernelSize; k++)
    {
        const int input_idx = window_base + k;
        if (input_idx >= 0 && input_idx < length)
            acc += inputValue(input_idx) * kernelValue(k);
    }
    return acc;
}

uint64_t pimBlockAddr(PIMKernel& kernel, unsigned chan, unsigned flat_bank, unsigned row,
                      unsigned col)
{
    const unsigned num_bank_groups = getConfigParam(UINT, "NUM_BANK_GROUPS");
    const unsigned banks_per_group = getConfigParam(UINT, "NUM_BANKS") / num_bank_groups;
    return kernel.pim_addr_mgr_->addrGenSafe(chan, 0, flat_bank / banks_per_group,
                                             flat_bank % banks_per_group, row, col);
}

void enqueueChecked(shared_ptr<MultiChannelMemorySystem>& mem, bool is_write, uint64_t addr,
                    const string& tag, BurstType* burst)
{
    while (!mem->addTransaction(is_write, addr, tag, burst)) mem->update();
}
}  // namespace

PIMConv1DBenchFixture::PIMConv1DBenchFixture()
{
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_conv1d_bench",
                                                     256 * 16);
    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
}

PIMConv1DBenchFixture::~PIMConv1DBenchFixture() {}

void PIMConv1DBenchFixture::run()
{
    while (pim_mem_->hasPendingTransactions()) pim_mem_->update();
}

void PIMConv1DBenchFixture::testConv1D(int length)
{
    ASSERT_GT(length, 0);
    ASSERT_GE(length + 2 * kPadding, kKernelSize) << "Conv1D output length must be positive";

    const int lanes = activeLanes();
    const unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    const unsigned num_rows = getConfigParam(UINT, "NUM_ROWS");
    const int output_length = (length + 2 * kPadding - kKernelSize) / kStride + 1;
    const uint64_t outputs_per_slot =
        static_cast<uint64_t>(num_chans) * num_pim_blocks * static_cast<uint64_t>(lanes);
    const uint64_t num_slots =
        (static_cast<uint64_t>(output_length) + outputs_per_slot - 1) / outputs_per_slot;

    ASSERT_LT((num_slots - 1) * kDataRowStride + 2, static_cast<uint64_t>(kResultRowBase));
    ASSERT_LT(kResultRowBase + num_slots, static_cast<uint64_t>(num_rows));

    cout << ">> Running Conv1D Bench | Size: " << length << " | Kernel: " << kKernelSize
         << " | Stride: " << kStride << " | Padding: " << kPadding
         << " | Output: " << output_length << endl;
    cout << "> Layout: channels=" << num_chans << " | PIM blocks/channel=" << num_pim_blocks
         << " | lanes/burst=" << lanes << " | output slots=" << num_slots << endl;

    vector<float> reference(output_length);
    for (int out_idx = 0; out_idx < output_length; out_idx++)
        reference[out_idx] = referenceConvAt(out_idx, length);

    const uint64_t total_preload_bursts =
        num_slots * 2 * kMacGroup * num_chans * num_pim_blocks;
    vector<BurstType> im2col_bursts(static_cast<size_t>(total_preload_bursts));
    uint64_t preloaded_bursts = 0;
    for (uint64_t slot = 0; slot < num_slots; slot++)
    {
        for (int group = 0; group < 2; group++)
        {
            const int tap_base = group * kMacGroup;
            const unsigned row = static_cast<unsigned>(slot * kDataRowStride + group * 2);
            for (int tap = 0; tap < kMacGroup; tap++)
            {
                for (unsigned chan = 0; chan < num_chans; chan++)
                {
                    for (unsigned pb = 0; pb < num_pim_blocks; pb++)
                    {
                        BurstType& burst = im2col_bursts[static_cast<size_t>(preloaded_bursts)];
                        for (int lane = 0; lane < lanes; lane++)
                        {
                            const uint64_t out_idx =
                                slot * outputs_per_slot +
                                (static_cast<uint64_t>(chan) * num_pim_blocks + pb) * lanes +
                                lane;
                            float value = 0.0f;
                            const int k = tap_base + tap;
                            if (out_idx < static_cast<uint64_t>(output_length) && k < kKernelSize)
                            {
                                const int input_idx =
                                    static_cast<int>(out_idx) * kStride - kPadding + k;
                                if (input_idx >= 0 && input_idx < length)
                                    value = inputValue(input_idx);
                            }
                            setLane(burst, lane, value);
                        }

                        const uint64_t addr =
                            pimBlockAddr(*kernel_, chan, pb * 2, row, static_cast<unsigned>(tap));
                        enqueueChecked(pim_mem_, true, addr, "CONV1D_IM2COL_PRELOAD", &burst);
                        preloaded_bursts++;
                    }
                }
            }
        }
    }
    run();
    cout << "> Preload complete: " << preloaded_bursts << " im2col bursts" << endl;

    vector<PIMCmd> mac_cmds;
    mac_cmds.push_back(PIMCmd(PIMCmdType::MAC, PIMOpdType::GRF_B, PIMOpdType::GRF_A,
                              PIMOpdType::EVEN_BANK, 0, 0, 0, 0));
    mac_cmds.push_back(PIMCmd(PIMCmdType::EXIT, 0));

    kernel_->setPIMControl(true, kernel_->getToggleCond(pimBankType::EVEN_BANK), false, false);
    kernel_->parkIn();
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    kernel_->programCrf(mac_cmds);
    kernel_->runPIM();
    cout << "> PIM setup complete" << endl;

    BurstType trigger;
    uint64_t mac_trigger_transactions = 0;
    uint64_t result_write_transactions = 0;
    const uint64_t start_cycle = kernel_->getCycle();

    for (uint64_t slot = 0; slot < num_slots; slot++)
    {
        for (int k = 0; k < kKernelSize; k++)
        {
            const int group = k / kMacGroup;
            const int tap = k % kMacGroup;
            BurstType weight = splatBurst(kernelValue(k));
            kernel_->addTransactionAll(true, 0, 1, kernel_->pim_reg_ra_, 0x8,
                                       "CONV1D_WEIGHT_TO_GRFA", &weight, true);
            kernel_->runPIM();

            kernel_->setPIMControl(true, kernel_->getToggleCond(pimBankType::EVEN_BANK), false,
                                   k == 0);
            kernel_->changePIMMode(dramMode::HAB, dramMode::HAB_PIM);
            kernel_->runPIM();

            const unsigned row = static_cast<unsigned>(slot * kDataRowStride + group * 2);
            kernel_->addTransactionAll(false, 0, 0, row, tap, "CONV1D_MAC", &trigger, true);
            mac_trigger_transactions += num_chans;
            kernel_->runPIM();

            if (k == 0)
            {
                kernel_->setPIMControl(true, kernel_->getToggleCond(pimBankType::EVEN_BANK),
                                       false, false);
            }
            kernel_->changePIMMode(dramMode::HAB_PIM, dramMode::HAB);
            kernel_->runPIM();
        }

        const unsigned result_row = static_cast<unsigned>(kResultRowBase + slot);
        kernel_->addTransactionAll(true, 0, 1, result_row, 0, "CONV1D_GRFB_TO_RESULT", &trigger,
                                   true);
        result_write_transactions += num_chans;
        kernel_->runPIM();
    }

    const uint64_t cycles = kernel_->getCycle() - start_cycle;
    cout << "> PIM compute complete" << endl;

    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    kernel_->runPIM();

    vector<float> actual(output_length, 0.0f);
    vector<BurstType> result_bursts(static_cast<size_t>(num_slots) * num_chans * num_pim_blocks);
    for (uint64_t slot = 0; slot < num_slots; slot++)
    {
        const unsigned result_row = static_cast<unsigned>(kResultRowBase + slot);
        for (unsigned chan = 0; chan < num_chans; chan++)
        {
            for (unsigned pb = 0; pb < num_pim_blocks; pb++)
            {
                const size_t result_idx =
                    static_cast<size_t>((slot * num_chans + chan) * num_pim_blocks + pb);
                const uint64_t addr = pimBlockAddr(*kernel_, chan, pb * 2 + 1, result_row, 0);
                enqueueChecked(pim_mem_, false, addr, "CONV1D_RESULT_READ",
                               &result_bursts[result_idx]);
            }
        }
    }
    run();

    for (uint64_t slot = 0; slot < num_slots; slot++)
    {
        for (unsigned chan = 0; chan < num_chans; chan++)
        {
            for (unsigned pb = 0; pb < num_pim_blocks; pb++)
            {
                const size_t result_idx =
                    static_cast<size_t>((slot * num_chans + chan) * num_pim_blocks + pb);

                for (int lane = 0; lane < lanes; lane++)
                {
                    const uint64_t out_idx =
                        slot * outputs_per_slot +
                        (static_cast<uint64_t>(chan) * num_pim_blocks + pb) * lanes + lane;
                    if (out_idx < static_cast<uint64_t>(output_length))
                        actual[static_cast<size_t>(out_idx)] =
                            getLane(result_bursts[result_idx], lane);
                }
            }
        }
    }

    double max_abs_error = 0.0;
    for (int out_idx = 0; out_idx < output_length; out_idx++)
    {
        const double err = fabs(static_cast<double>(actual[out_idx] - reference[out_idx]));
        max_abs_error = max(max_abs_error, err);
    }
    EXPECT_LE(max_abs_error, 0.05);
    cout << "> Check sample: actual[0]=" << actual[0] << " reference[0]=" << reference[0]
         << " actual[last]=" << actual.back() << " reference[last]=" << reference.back() << endl;

    kernel_->parkOut();
    kernel_->runPIM();

    const double time_ms = static_cast<double>(cycles) * tCK * 1e-6;
    const uint64_t logical_mac_ops = static_cast<uint64_t>(output_length) * kKernelSize;
    cout << "> Results: logical MAC ops: " << logical_mac_ops
         << " | im2col bursts: " << preloaded_bursts
         << " | MAC trigger transactions: " << mac_trigger_transactions
         << " | result write transactions: " << result_write_transactions
         << " | Cycles: " << cycles << " | Time: " << time_ms
         << " ms | MaxAbsError: " << max_abs_error << endl;
}

TEST_F(PIMConv1DBenchFixture, conv1d_10K) { testConv1D(10 * 1024); }
TEST_F(PIMConv1DBenchFixture, conv1d_100K) { testConv1D(100 * 1024); }
TEST_F(PIMConv1DBenchFixture, conv1d_1M) { testConv1D(1024 * 1024); }

// ./sim --gtest_filter=PIMConv1DBenchFixture.conv1d_1M
