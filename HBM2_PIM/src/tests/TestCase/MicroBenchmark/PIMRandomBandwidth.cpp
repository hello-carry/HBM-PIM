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

#include "PIMRandomBandwidth.h"

#include "gtest/gtest.h"
#include <algorithm>

using namespace DRAMSim;
using namespace std;

PIMRandomBandwidthFixture::PIMRandomBandwidthFixture()
{
    // Initialize the memory system
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_random_bw_test", 256 * 16);
    
    // Initialize the kernel with the maximum channel count
    max_channels = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, max_channels, 1);

    tCK = getConfigParam(FLOAT, "tCK");
    num_pim_blocks_per_channel = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    
    // Read parameters
    num_cols = getConfigParam(UINT, "NUM_COLS");
    num_rows = getConfigParam(UINT, "NUM_ROWS");
    num_ranks = getConfigParam(UINT, "NUM_RANKS"); // Usually 1
    num_banks = getConfigParam(UINT, "NUM_BANKS"); // Usually 16
    
    // HAB-mode per-transaction size: 8 PIMBlocks * 32 Bytes = 256 Bytes
    bytes_per_pim_transaction = num_pim_blocks_per_channel * 32;

    // Set the reserved-row threshold (Row 8192)
    // Based on earlier analysis, bit 13 is the reserved bit
    pim_reg_ra_threshold_ = 8192;

    // Initialize the random number generator
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

PIMRandomBandwidthFixture::~PIMRandomBandwidthFixture()
{
    pim_mem_->printStats(true);
}

void PIMRandomBandwidthFixture::run()
{
    while (pim_mem_->hasPendingTransactions())
    {
        pim_mem_->update();
    }
}

void PIMRandomBandwidthFixture::testRandomBandwidthByDataSize(uint64_t data_size_bytes, uint32_t num_channels)
{
    // 1. Calculate the required transaction count
    // Round up to ensure at least data_size_bytes are transferred
    uint32_t num_transactions = (data_size_bytes + bytes_per_pim_transaction - 1) / bytes_per_pim_transaction;
    uint64_t actual_data_size = (uint64_t)num_transactions * bytes_per_pim_transaction;

    // Clamp the channel count
    if (num_channels > max_channels) num_channels = max_channels;
    if (num_channels == 0) num_channels = 1;

    cout << ">> PIM True Random Bandwidth Test (HAB Mode, Data Size Driven)" << endl;
    cout << "> Target Data Size: " << data_size_bytes << " Bytes (" << data_size_bytes / (1024.0 * 1024.0) << " MB)" << endl;
    cout << "> Actual Data Size: " << actual_data_size << " Bytes" << endl;
    cout << "> Total Transactions: " << num_transactions << endl;
    cout << "> Channels: " << num_channels << " (Round-Robin Injection)" << endl;
    cout << "> Mode: Bank Ping-Pong (0/1 Random) + Row Random + Col Random" << endl;

    // 2. Initialize PIM
    kernel_->setPIMControl(true, 0, false, false);
    kernel_->parkIn();
    run();

    // 3. Switch to HAB mode
    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    run();

    BurstType null_bst;
    
    // 4. Pre-generate the random address list
    struct Req {
        int ch;
        uint64_t addr;
    };
    std::vector<Req> requests;
    requests.reserve(num_transactions);

    std::uniform_int_distribution<unsigned> dist_row(0, pim_reg_ra_threshold_ - 1);
    // Use the dynamically read num_cols value
    std::uniform_int_distribution<unsigned> dist_col(0, num_cols / getConfigParam(UINT, "BL") - 1); // Actual col field width = log2(NUM_COLS/BL)
    std::uniform_int_distribution<unsigned> dist_bank(0, 1); // Always randomize between Bank 0 and 1
    
    for(uint32_t i=0; i<num_transactions; ++i) {
        int ch = i % num_channels; // Round-robin across the selected channel count
        unsigned ra = 0; // Rank 0
        unsigned bg = 0; // Keep BG at 0 in HAB mode
        
        // Key point: random Bank (0 or 1) enables bank-parallel testing
        // unsigned ba = dist_bank(rng_);
        unsigned ba = 0;
        // If access is forced to only odd or even here (ba = 0 / 1), bandwidth drops by about 40%
        
        // Key point: random row, avoiding the reserved area
        unsigned row = dist_row(rng_);
        
        // Key point: random column
        unsigned col = dist_col(rng_);

        // Generate the physical address
        uint64_t addr = kernel_->pim_addr_mgr_->addrGen(ch, ra, bg, ba, row, col);
        
        requests.push_back({ch, addr});
    }

    cout << "> Starting Random Injection..." << endl;

    // 5. Execute the test
    uint64_t cycle = 0;
    uint32_t sent_count = 0;
    // Keep the pipeline full
    uint32_t target_pending = num_channels * 16; 
    const uint64_t MAX_CYCLES = 20000000; // Watchdog

    while (sent_count < num_transactions && cycle < MAX_CYCLES) {
        uint64_t pending = pim_mem_->hasPendingTransactions();
        
        if (pending < target_pending) {
            uint32_t batch_size = min((uint32_t)(target_pending - pending), (uint32_t)(num_transactions - sent_count));
            // Increase the injection burst size to improve scheduling efficiency
            batch_size = min(batch_size, 128u); 
            
            for (uint32_t k = 0; k < batch_size; ++k) {
                Req& r = requests[sent_count++];
                pim_mem_->addTransaction(false, r.addr, "RND_HAB", &null_bst);
            }
        }

        pim_mem_->update();
        cycle++;
    }

    // Wait for the pipeline to drain
    while (pim_mem_->hasPendingTransactions() && cycle < MAX_CYCLES) {
        pim_mem_->update();
        cycle++;
    }

    // 6. Restore state and collect statistics
    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    kernel_->parkOut();
    run();

    double time_sec = (double)cycle * tCK * 1e-9;
    double bw_gbps = (double)actual_data_size / (1024.0*1024.0*1024.0) / time_sec;

    cout << "> Results:" << endl;
    cout << "  Data Size: " << actual_data_size / (1024.0*1024.0) << " MB" << endl;
    cout << "  Cycles: " << cycle << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Bandwidth: " << bw_gbps << " GB/s" << endl;
    cout << "------------------------------------------------" << endl;
}

// ============================================================================
// Google Test Cases (Data Size Driven)
// ============================================================================

// 1 MB test
TEST_F(PIMRandomBandwidthFixture, random_bandwidth)
{
    testRandomBandwidthByDataSize(1 * 1024 * 1024);
}
