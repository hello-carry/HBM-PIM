/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#include "PIMHist.h"
#include "gtest/gtest.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using namespace DRAMSim;
using namespace std;

namespace
{
constexpr uint64_t kValuesPerBurst = 8;  // 32B burst / 4B uint32_t
constexpr unsigned kInputRowBase = 0;
constexpr unsigned kLocalHistRowBase = 1024;
constexpr unsigned kGlobalHistRowBase = 2048;
constexpr uint64_t kUpdateChunkBursts = 8192;
constexpr uint64_t kProgressElements = 1000000;

uint64_t addrFromFlatBank(const shared_ptr<PIMKernel>& kernel, unsigned chan, unsigned flat_bank,
                          unsigned row, unsigned col)
{
    PIMAddrManager* addr_mgr = kernel->pim_addr_mgr_.get();
    const unsigned banks_per_group = addr_mgr->num_banks_ / addr_mgr->num_bank_groups_;
    return addr_mgr->addrGen(chan, 0, flat_bank / banks_per_group, flat_bank % banks_per_group,
                             row, col);
}

uint64_t shardCount(const shared_ptr<PIMKernel>& kernel)
{
    return static_cast<uint64_t>(kernel->pim_addr_mgr_->num_pim_chans_) *
           getConfigParam(UINT, "NUM_PIM_BLOCKS");
}

unsigned shardChan(unsigned shard)
{
    return shard % getConfigParam(UINT, "NUM_CHANS");
}

unsigned shardPimBlock(unsigned shard)
{
    return shard / getConfigParam(UINT, "NUM_CHANS");
}

uint64_t inputAddr(const shared_ptr<PIMKernel>& kernel, uint64_t input_burst)
{
    const uint64_t shards = shardCount(kernel);
    const unsigned shard = static_cast<unsigned>(input_burst % shards);
    const uint64_t shard_slot = input_burst / shards;
    const unsigned row =
        kInputRowBase + static_cast<unsigned>(shard_slot / kernel->pim_addr_mgr_->num_cols_per_bl_);
    const unsigned col = static_cast<unsigned>(shard_slot % kernel->pim_addr_mgr_->num_cols_per_bl_);
    return addrFromFlatBank(kernel, shardChan(shard), shardPimBlock(shard) * 2, row, col);
}

uint64_t localHistAddr(const shared_ptr<PIMKernel>& kernel, unsigned shard, uint64_t bin_burst)
{
    const unsigned row =
        kLocalHistRowBase +
        static_cast<unsigned>(bin_burst / kernel->pim_addr_mgr_->num_cols_per_bl_);
    const unsigned col = static_cast<unsigned>(bin_burst % kernel->pim_addr_mgr_->num_cols_per_bl_);
    return addrFromFlatBank(kernel, shardChan(shard), shardPimBlock(shard) * 2 + 1, row, col);
}

uint64_t globalHistAddr(const shared_ptr<PIMKernel>& kernel, uint64_t bin_burst)
{
    const unsigned num_chans = getConfigParam(UINT, "NUM_CHANS");
    const unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    const unsigned chan = static_cast<unsigned>(bin_burst % num_chans);
    const unsigned pb = static_cast<unsigned>((bin_burst / num_chans) % num_pim_blocks);
    const uint64_t slot = bin_burst / (static_cast<uint64_t>(num_chans) * num_pim_blocks);
    const unsigned row =
        kGlobalHistRowBase + static_cast<unsigned>(slot / kernel->pim_addr_mgr_->num_cols_per_bl_);
    const unsigned col = static_cast<unsigned>(slot % kernel->pim_addr_mgr_->num_cols_per_bl_);
    return addrFromFlatBank(kernel, chan, pb * 2, row, col);
}

void enqueueChecked(const shared_ptr<MultiChannelMemorySystem>& mem, bool is_write, uint64_t addr,
                    BurstType* burst)
{
    while (!mem->addTransaction(is_write, addr, burst)) mem->update();
}
}  // namespace

PIMHistBenchFixture::PIMHistBenchFixture()
{
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_hist_bench",
                                                     256 * 16);
    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
}

PIMHistBenchFixture::~PIMHistBenchFixture() {}

void PIMHistBenchFixture::run()
{
    while (pim_mem_->hasPendingTransactions()) pim_mem_->update();
}

void PIMHistBenchFixture::testHist(uint64_t num_elements, uint32_t num_bins)
{
    ASSERT_GT(num_elements, 0u);
    ASSERT_GT(num_bins, 0u);
    ASSERT_EQ(num_bins % kValuesPerBurst, 0u) << "bin counters are packed as uint32 lanes";

    cout << ">> MacroBench: Histogram (striped input, private local hist + reduction)" << endl;
    cout << "> Channels: " << num_chans << " | Elements: " << num_elements
         << " | Bins: " << num_bins << endl;

    vector<uint32_t> input(num_elements);
    vector<uint32_t> hist_host(num_bins, 0);

    mt19937 rng(0);
    uniform_int_distribution<uint32_t> dist(0, num_bins - 1);
    for (uint64_t i = 0; i < num_elements; i++)
    {
        input[i] = dist(rng);
        hist_host[input[i]]++;
    }

    uint64_t golden_total = accumulate(hist_host.begin(), hist_host.end(), uint64_t{0});
    ASSERT_EQ(golden_total, num_elements);

    const uint64_t input_bursts = (num_elements + kValuesPerBurst - 1) / kValuesPerBurst;
    const uint64_t hist_bursts = (num_bins + kValuesPerBurst - 1) / kValuesPerBurst;
    const uint64_t shards = shardCount(kernel_);
    const uint64_t input_slots_per_shard = (input_bursts + shards - 1) / shards;
    const uint64_t input_rows =
        (input_slots_per_shard + kernel_->pim_addr_mgr_->num_cols_per_bl_ - 1) /
        kernel_->pim_addr_mgr_->num_cols_per_bl_;
    const uint64_t hist_rows =
        (hist_bursts + kernel_->pim_addr_mgr_->num_cols_per_bl_ - 1) /
        kernel_->pim_addr_mgr_->num_cols_per_bl_;

    ASSERT_LT(kInputRowBase + input_rows, static_cast<uint64_t>(kLocalHistRowBase));
    ASSERT_LT(kLocalHistRowBase + hist_rows, static_cast<uint64_t>(kGlobalHistRowBase));
    ASSERT_LT(kGlobalHistRowBase + hist_rows, static_cast<uint64_t>(1u << 13))
        << "PIM reserved rows start at row bit 13";
    ASSERT_LT(kGlobalHistRowBase + hist_rows, static_cast<uint64_t>(kernel_->pim_addr_mgr_->num_rows_));

    cout << "> Layout: shards=" << shards << " | input bursts=" << input_bursts
         << " | hist bursts/shard=" << hist_bursts << endl;

    vector<BurstType> input_burst_data(input_bursts);
    for (uint64_t burst = 0; burst < input_bursts; burst++)
    {
        for (uint64_t lane = 0; lane < kValuesPerBurst; lane++)
        {
            const uint64_t elem = burst * kValuesPerBurst + lane;
            input_burst_data[burst].u32Data_[lane] = (elem < num_elements) ? input[elem] : 0;
        }
    }

    vector<BurstType> local_hist_bursts(static_cast<size_t>(shards * hist_bursts));
    vector<BurstType> global_hist_bursts(static_cast<size_t>(hist_bursts));

    cout << "> Preloading input and zero counters" << endl;
    for (uint64_t burst = 0; burst < input_bursts; burst++)
        enqueueChecked(pim_mem_, true, inputAddr(kernel_, burst), &input_burst_data[burst]);
    for (uint64_t shard = 0; shard < shards; shard++)
    {
        for (uint64_t bin_burst = 0; bin_burst < hist_bursts; bin_burst++)
        {
            BurstType& counter_burst =
                local_hist_bursts[static_cast<size_t>(shard * hist_bursts + bin_burst)];
            enqueueChecked(pim_mem_, true,
                           localHistAddr(kernel_, static_cast<unsigned>(shard), bin_burst),
                           &counter_burst);
        }
    }
    for (uint64_t bin_burst = 0; bin_burst < hist_bursts; bin_burst++)
        enqueueChecked(pim_mem_, true, globalHistAddr(kernel_, bin_burst),
                       &global_hist_bursts[bin_burst]);
    run();
    cout << "> Preload complete" << endl;

    uint64_t input_reads = 0;
    uint64_t local_counter_reads = 0;
    uint64_t local_counter_writes = 0;
    uint64_t next_progress = kProgressElements;

    uint64_t update_start = kernel_->getCycle();
    for (uint64_t chunk_begin = 0; chunk_begin < input_bursts; chunk_begin += kUpdateChunkBursts)
    {
        const uint64_t chunk_end = min(input_bursts, chunk_begin + kUpdateChunkBursts);
        const uint64_t chunk_bursts = chunk_end - chunk_begin;

        vector<BurstType> input_read_sinks;
        vector<BurstType> counter_read_sinks;
        vector<BurstType> counter_write_bursts;
        input_read_sinks.reserve(static_cast<size_t>(chunk_bursts));
        counter_read_sinks.reserve(static_cast<size_t>(chunk_bursts * kValuesPerBurst));
        counter_write_bursts.reserve(static_cast<size_t>(chunk_bursts * kValuesPerBurst));

        for (uint64_t burst = chunk_begin; burst < chunk_end; burst++)
        {
            input_read_sinks.emplace_back();
            enqueueChecked(pim_mem_, false, inputAddr(kernel_, burst), &input_read_sinks.back());
            input_reads++;

            const unsigned shard = static_cast<unsigned>(burst % shards);
            array<uint32_t, kValuesPerBurst> unique_bin_bursts{};
            array<array<uint32_t, kValuesPerBurst>, kValuesPerBurst> lane_increments{};
            unsigned unique_count = 0;

            for (uint64_t lane = 0; lane < kValuesPerBurst; lane++)
            {
                const uint64_t elem = burst * kValuesPerBurst + lane;
                if (elem >= num_elements) break;

                const uint32_t bin = input_burst_data[burst].u32Data_[lane];
                ASSERT_LT(bin, num_bins);
                const uint32_t bin_burst = bin / kValuesPerBurst;
                const uint32_t bin_lane = bin % kValuesPerBurst;

                unsigned unique_idx = 0;
                for (; unique_idx < unique_count; unique_idx++)
                {
                    if (unique_bin_bursts[unique_idx] == bin_burst) break;
                }
                if (unique_idx == unique_count)
                {
                    unique_bin_bursts[unique_count++] = bin_burst;
                }
                lane_increments[unique_idx][bin_lane]++;
            }

            for (unsigned u = 0; u < unique_count; u++)
            {
                const uint32_t bin_burst = unique_bin_bursts[u];
                const size_t local_idx =
                    static_cast<size_t>(shard * hist_bursts + bin_burst);
                const uint64_t counter_addr = localHistAddr(kernel_, shard, bin_burst);

                counter_read_sinks.emplace_back();
                enqueueChecked(pim_mem_, false, counter_addr, &counter_read_sinks.back());
                local_counter_reads++;

                for (uint64_t lane = 0; lane < kValuesPerBurst; lane++)
                    local_hist_bursts[local_idx].u32Data_[lane] += lane_increments[u][lane];

                counter_write_bursts.push_back(local_hist_bursts[local_idx]);
                enqueueChecked(pim_mem_, true, counter_addr, &counter_write_bursts.back());
                local_counter_writes++;
            }
        }

        kernel_->runPIM();

        const uint64_t processed_elements = min(num_elements, chunk_end * kValuesPerBurst);
        if (processed_elements >= next_progress)
        {
            cout << "  Update progress: " << processed_elements << " / " << num_elements
                 << " elements" << endl;
            while (next_progress <= processed_elements) next_progress += kProgressElements;
        }
    }
    uint64_t update_cycles = kernel_->getCycle() - update_start;

    uint64_t reduction_reads = 0;
    uint64_t reduction_writes = 0;
    uint64_t reduction_start = kernel_->getCycle();

    vector<BurstType> reduction_read_sinks(local_hist_bursts.size());
    for (uint64_t shard = 0; shard < shards; shard++)
    {
        for (uint64_t bin_burst = 0; bin_burst < hist_bursts; bin_burst++)
        {
            const size_t idx = static_cast<size_t>(shard * hist_bursts + bin_burst);
            enqueueChecked(pim_mem_, false,
                           localHistAddr(kernel_, static_cast<unsigned>(shard), bin_burst),
                           &reduction_read_sinks[idx]);
            reduction_reads++;

            for (uint64_t lane = 0; lane < kValuesPerBurst; lane++)
                global_hist_bursts[bin_burst].u32Data_[lane] += local_hist_bursts[idx].u32Data_[lane];
        }
    }
    kernel_->runPIM();

    vector<BurstType> global_write_bursts = global_hist_bursts;
    for (uint64_t bin_burst = 0; bin_burst < hist_bursts; bin_burst++)
    {
        enqueueChecked(pim_mem_, true, globalHistAddr(kernel_, bin_burst),
                       &global_write_bursts[bin_burst]);
        reduction_writes++;
    }
    kernel_->runPIM();
    uint64_t reduction_cycles = kernel_->getCycle() - reduction_start;

    vector<BurstType> global_readback(hist_bursts);
    for (uint64_t bin_burst = 0; bin_burst < hist_bursts; bin_burst++)
        enqueueChecked(pim_mem_, false, globalHistAddr(kernel_, bin_burst),
                       &global_readback[bin_burst]);
    run();

    uint32_t mismatch_bins = 0;
    uint64_t max_abs_error = 0;
    for (uint32_t bin = 0; bin < num_bins; bin++)
    {
        const uint32_t actual = global_readback[bin / kValuesPerBurst].u32Data_[bin % kValuesPerBurst];
        const uint32_t expected = hist_host[bin];
        if (actual != expected)
        {
            mismatch_bins++;
            const uint64_t err = (actual > expected) ? (actual - expected) : (expected - actual);
            max_abs_error = max(max_abs_error, err);
        }
    }
    EXPECT_EQ(mismatch_bins, 0u);

    const uint64_t total_kernel_cycles = update_cycles + reduction_cycles;
    double update_ms = (double)update_cycles * tCK * 1e-6;
    double reduction_ms = (double)reduction_cycles * tCK * 1e-6;
    double total_ms = (double)total_kernel_cycles * tCK * 1e-6;

    cout << "> Results:" << endl;
    cout << "  Input Burst Reads: " << input_reads << endl;
    cout << "  Local Counter Reads: " << local_counter_reads << endl;
    cout << "  Local Counter Writes: " << local_counter_writes << endl;
    cout << "  Reduction Reads: " << reduction_reads << endl;
    cout << "  Reduction Writes: " << reduction_writes << endl;
    cout << "  Total Updates: " << golden_total << endl;
    cout << "  Update Cycles: " << update_cycles << " (" << update_ms << " ms)" << endl;
    cout << "  Reduction Cycles: " << reduction_cycles << " (" << reduction_ms << " ms)" << endl;
    cout << "  Total Kernel Cycles: " << total_kernel_cycles << " (" << total_ms << " ms)" << endl;
    cout << "  Mismatch Bins: " << mismatch_bins << " | MaxBinError: " << max_abs_error << endl;
    cout << "------------------------------------------------" << endl;
}

TEST_F(PIMHistBenchFixture, hist_1M_256) { testHist(1000000, 256); }
TEST_F(PIMHistBenchFixture, hist_5M_256) { testHist(5 * 1000000, 256); }
TEST_F(PIMHistBenchFixture, hist_10M_256) { testHist(10 * 1000000, 256); }
