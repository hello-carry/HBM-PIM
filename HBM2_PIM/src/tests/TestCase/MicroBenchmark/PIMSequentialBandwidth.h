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

#ifndef __PIM_SEQUENTIAL_BANDWIDTH_TEST_CASE_H__
#define __PIM_SEQUENTIAL_BANDWIDTH_TEST_CASE_H__

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMSequentialBandwidthFixture: fully sequential access bandwidth test
 * (automatically crosses channels and rows)
 *
 * Description:
 * - Use contiguous addresses to test Bank->GRF sequential memory-access bandwidth
 * - Core idea: the control variable is data size, not channel count
 * - Automatically cross channels and rows by generating Scheme8 contiguous
 *   addresses using the current system channel count
 *
 * Scheme8 address mapping (high to low): rank -> row -> col -> bankgroup -> bank -> chan -> offset
 * When addresses grow contiguously: chan -> bank -> bankgroup -> col -> row -> rank
 *
 * HAB-mode mechanism (single-channel level):
 * - Each READ transaction triggers 8 PIMBlocks in one channel to read in parallel
 * - Each PIMBlock reads 32 bytes from a different bank
 * - Per-transaction data size: 8 PIMBlocks x 32 bytes = 256 bytes/channel
 * - Row size: 128 cols x 8 bytes = 1024 bytes = 1 KB
 *
 * Data size versus row/channel crossing (bounded by NUM_CHANS in the current
 * system_hbm.ini):
 * - 256 bytes (1 transaction, 1 ch): no row crossing
 * - 512 bytes (2 transactions, 2 ch): no row crossing
 * - 1 KB (4 transactions, 4 ch): no row crossing
 * - 4 KB (16 transactions, 16 ch if available): use all current 16 channels
 * - 8 KB+: increase sequential column accesses on all available channels,
 *   crossing rows when necessary
 *
 * Test strategy:
 * - Start at 256 B and gradually increase: 512 B, 1 KB, 2 KB, 4 KB, 8 KB, 16 KB, ...
 * - Generate addresses in Scheme8 low-bit channel-interleaved order
 * - Observe bandwidth changes under different data sizes
 */
namespace DRAMSim
{
class PIMSequentialBandwidthFixture : public testing::Test
{
  public:
    PIMSequentialBandwidthFixture()
    {
        // Create the HBM2-PIM system; channel count comes from system_hbm.ini
        pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                        "system_hbm.ini", ".", "pim_seq_bw_test", 256 * 16);

        tCK = getConfigParam(FLOAT, "tCK");
        max_channels = getConfigParam(UINT, "NUM_CHANS");
        num_pim_blocks_per_channel = getConfigParam(UINT, "NUM_PIM_BLOCKS");  // 8
        bytes_per_pimblock = 16 * 2;  // 32 bytes
        bytes_per_transaction = (getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8) *
                               getConfigParam(UINT, "BL");  // 32 bytes (CPU view)

        // Minimum HAB-mode granularity: one transaction in one channel triggers 8 PIMBlocks
        bytes_per_pim_transaction = num_pim_blocks_per_channel * bytes_per_pimblock;  // 256 bytes

        // Row size, used for row-crossing analysis
        num_cols_per_bl = getConfigParam(UINT, "NUM_COLS");  // 128
        bytes_per_row = num_cols_per_bl * 8;  // 1024 bytes = 1 KB
    }

    virtual ~PIMSequentialBandwidthFixture()
    {
        pim_mem_->printStats(true);
    }

    // Data-size controlled sequential memory-access bandwidth test (uses HAB mode and sends addresses directly)
    // @param data_bytes_per_trigger: data transferred per trigger, in bytes; automatically aligned to 256B
    // @param num_triggers: trigger count (default 512)
    //
    // Core idea:
    // - The control variable is data size, not channel count
    // - Fill available channels according to Scheme8 channel bits first, then
    //   increment column addresses sequentially within each channel
    // - addrGenSafe automatically handles row crossing after col overflow
    void testBandwidthByDataSize(uint32_t data_bytes_per_trigger,
                                  uint32_t num_triggers = 512)
    {
        // Calculate the required transaction count, automatically aligned to 256B granularity
        uint32_t requested_transactions =
            (data_bytes_per_trigger + bytes_per_pim_transaction - 1) / bytes_per_pim_transaction;

        // The channel count is determined by data size and capped by the actual NUM_CHANS.
        uint32_t channels_to_use = min((uint32_t)max_channels, requested_transactions);
        channels_to_use = max(channels_to_use, 1u);

        // Sequential transactions per channel. After the channel count is exceeded,
        // continue incrementing column addresses and let addrGenSafe cross rows.
        uint32_t transactions_per_channel =
            (requested_transactions + channels_to_use - 1) / channels_to_use;
        uint32_t transactions_per_trigger = channels_to_use * transactions_per_channel;

        // Reset sequential address state for this test
        seq_col_ = 0;
        seq_row_ = 0;
        channels_to_use_ = channels_to_use;
        trigger_count_ = 0;
        addr_log_.open("seq_bdw.txt", ios::app);
        addr_log_ << "\n# === SeqBW: data=" << data_bytes_per_trigger
                  << "B  triggers=" << num_triggers
                  << "  channels=" << channels_to_use
                  << "  txn/ch=" << transactions_per_channel << " ===\n";
        addr_log_ << "# t=trigger  ch  [start_row,col -> end_row,col]  phys_addr\n";

        // Actual data size, which can be larger than requested because of alignment
        uint32_t actual_bytes_per_trigger = transactions_per_trigger * bytes_per_pim_transaction;
        uint32_t total_bytes = actual_bytes_per_trigger * num_triggers;

        cout << ">> PIM Sequential Bandwidth Test (HAB Mode, Address-Driven)" << endl;
        cout << "> Requested data per trigger: " << data_bytes_per_trigger << " bytes" << endl;
        cout << "> Actual data per trigger: " << actual_bytes_per_trigger
             << " bytes (" << (actual_bytes_per_trigger / 1024.0) << " KB)" << endl;
        cout << "> Channels used: " << channels_to_use << " / " << max_channels << endl;
        cout << "> Transactions per channel: " << transactions_per_channel << endl;
        cout << "> Transactions per trigger: " << transactions_per_trigger
             << " (" << channels_to_use << " ch × " << transactions_per_channel << ")" << endl;
        cout << "> Total data: " << total_bytes
             << " bytes (" << (total_bytes / (1024.0 * 1024.0)) << " MB)" << endl;
        cout << "> Num triggers: " << num_triggers << endl;

        // Analyze the memory-access pattern
        analyzeAccessPattern(channels_to_use, transactions_per_channel);

        // Create the kernel for HAB mode
        kernel_ = make_shared<PIMKernel>(pim_mem_, channels_to_use, 1);

        // PIM initialization flow
        kernel_->setPIMControl(true, 0, false, false);
        cout << "> PIM control set" << endl;

        kernel_->parkIn();
        run();

        kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
        run();

        cout << "> Using HAB mode with sequential addresses" << endl;

        cout << "> Adding " << num_triggers << " trigger(s)..." << endl;

        // Pipeline parameters
        uint64_t target_pending_low = 64 * (channels_to_use / 8 + 1);
        uint64_t target_pending_high = target_pending_low * 2;
        const uint64_t MAX_CYCLES = 2000000;
        uint64_t cycle = 0;
        uint32_t triggers_added = 0;

        cout << "> Using pipeline mode (keep pending between " << target_pending_low
             << " and " << target_pending_high << ")" << endl;
        cout << "> Starting execution..." << endl;

        // Pipeline processing
        while (triggers_added < num_triggers) {
            uint64_t pending = pim_mem_->hasPendingTransactions();

            if (pending < target_pending_low && triggers_added < num_triggers) {
                uint32_t space_available = (target_pending_high - pending) / transactions_per_trigger;
                uint32_t to_add = min(space_available, num_triggers - triggers_added);
                to_add = max(to_add, 1u);

                for (uint32_t t = 0; t < to_add; t++) {
                    addSequentialAddresses(transactions_per_channel);
                    triggers_added++;
                }
            }

            pim_mem_->update();
            cycle++;

            if (cycle % 5000 == 0) {
                cout << "> Cycle " << cycle << ": triggers_added=" << triggers_added
                     << "/" << num_triggers << ", pending=" << pim_mem_->hasPendingTransactions() << endl;
            }

            if (cycle >= MAX_CYCLES) {
                cout << "> ERROR: Reached MAX_CYCLES limit!" << endl;
                break;
            }
        }

        // Wait for the remaining transactions to complete
        while (pim_mem_->hasPendingTransactions() && cycle < MAX_CYCLES) {
            pim_mem_->update();
            cycle++;
        }

        cout << "> Cycles executed: " << cycle << endl;

        // Calculate bandwidth
        double time_sec = (double)cycle * tCK * 1e-9;
        double bandwidth_gb_per_sec = (double)total_bytes / (1024.0 * 1024.0 * 1024.0) / time_sec;
        double bandwidth_mb_per_sec = (double)total_bytes / (1024.0 * 1024.0) / time_sec;

        uint32_t total_transactions = transactions_per_trigger * num_triggers;

        cout << "> Total transactions: " << total_transactions
             << " (" << num_triggers << " × " << transactions_per_trigger << ")" << endl;
        cout << "> Data per trigger: " << actual_bytes_per_trigger
             << " bytes (" << (actual_bytes_per_trigger / 1024.0) << " KB)" << endl;
        cout << "> Total data: " << total_bytes
             << " bytes (" << (total_bytes / (1024.0 * 1024.0)) << " MB)" << endl;
        cout << "> Cycle: " << cycle << endl;
        cout << "> Note: Addresses are issued in Scheme8 channel-interleaved order" << endl;

        // Choose the appropriate unit based on bandwidth size
        if (bandwidth_gb_per_sec >= 1.0) {
            cout << "> Bandwidth (Sequential, Auto-cross-channel): " << bandwidth_gb_per_sec << " GB/s" << endl;
        } else {
            cout << "> Bandwidth (Sequential, Auto-cross-channel): " << bandwidth_mb_per_sec << " MB/s" << endl;
        }
        cout << "> Time: " << (time_sec * 1000.0) << " ms" << endl;
        cout << "> Theoretical peak (per channel): ~57.8 GB/s" << endl;
        cout << "> Theoretical peak (" << channels_to_use << " ch): ~"
             << (57.8 * channels_to_use) << " GB/s" << endl;
        cout << endl;

        addr_log_.close();
    }

    // Add transactions for contiguous addresses (HAB mode, cross-channel access)
    // Each call starts sending transactions from seq_row_/seq_col_ and advances
    // the address afterward for the next call.
    void addSequentialAddresses(uint32_t transactions_per_channel)
    {
        // Log the address range this trigger will access, one line per channel
        for (uint32_t ch = 0; ch < channels_to_use_; ch++) {
            unsigned log_row = seq_row_;
            unsigned log_col = seq_col_;
            uint64_t start_addr =
                kernel_->pim_addr_mgr_->addrGenSafe(ch, 0, 0, 0, log_row, log_col);

            unsigned end_row = seq_row_;
            unsigned end_col = seq_col_ + transactions_per_channel - 1;
            while (end_col >= num_cols_per_bl) {
                end_col -= num_cols_per_bl;
                end_row++;
            }

            addr_log_ << "t=" << setw(4) << trigger_count_
                      << "  ch=" << setw(2) << ch
                      << "  [" << setw(4) << log_row << "," << setw(3) << log_col
                      << " -> " << setw(4) << end_row << "," << setw(3) << end_col << "]"
                      << "  0x" << hex << setw(12) << setfill('0') << start_addr
                      << dec << setfill(' ') << "\n";
        }
        trigger_count_++;

        // Issue read transactions starting from current sequential address
        kernel_->addTransactionAll(false, 0, 0, seq_row_, seq_col_,
                                   "SEQ_READ", &null_bst_, false,
                                   transactions_per_channel);

        // Advance address for next trigger (mirrors addrGenSafe wrapping logic)
        seq_col_ += transactions_per_channel;
        while (seq_col_ >= num_cols_per_bl) {
            seq_col_ -= num_cols_per_bl;
            seq_row_++;
        }
    }

    // Analyze the memory-access pattern
    void analyzeAccessPattern(uint32_t channels_to_use, uint32_t transactions_per_channel)
    {
        cout << "> Access pattern analysis:" << endl;

        // Each transaction is 256 bytes
        uint32_t num_transactions = channels_to_use * transactions_per_channel;
        uint32_t data_size = num_transactions * bytes_per_pim_transaction;

        // Estimate row crossing
        uint32_t bytes_per_channel = transactions_per_channel * bytes_per_pim_transaction;
        uint32_t rows_per_channel = (bytes_per_channel + bytes_per_row - 1) / bytes_per_row;

        cout << "  - Data size: " << data_size << " bytes" << endl;
        cout << "  - Channels used: " << channels_to_use << " / " << max_channels << endl;
        cout << "  - Transactions per channel: " << transactions_per_channel << endl;
        cout << "  - Estimated rows per channel: ~" << rows_per_channel
             << " (row size = " << bytes_per_row << " bytes)" << endl;
        cout << "  - Pattern: Scheme8 channel-interleaved sequential addresses" << endl;

        if (channels_to_use == 1) {
            cout << "  - Expected: Single-channel sequential access" << endl;
        } else if (channels_to_use < max_channels) {
            cout << "  - Expected: Partial-channel interleaving" << endl;
        } else {
            cout << "  - Expected: All available channels, sequential columns per channel" << endl;
        }
    }

    void run()
    {
        while (pim_mem_->hasPendingTransactions())
        {
            pim_mem_->update();
        }
    }

  protected:
    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    BurstType null_bst_;
    float tCK;
    unsigned max_channels;
    unsigned num_pim_blocks_per_channel;
    unsigned bytes_per_pimblock;
    unsigned bytes_per_transaction;
    unsigned bytes_per_pim_transaction;  // 256 bytes (8 PIMBlocks × 32B)
    unsigned num_cols_per_bl;            // 128 columns per row
    unsigned bytes_per_row;              // 1024 bytes (row size)

    // Sequential address tracking (persist across triggers)
    uint32_t seq_col_;
    uint32_t seq_row_;
    uint32_t channels_to_use_;
    uint32_t trigger_count_;
    ofstream addr_log_;
};

}  // namespace DRAMSim

#endif /*__PIM_SEQUENTIAL_BANDWIDTH_TEST_CASE_H__*/
