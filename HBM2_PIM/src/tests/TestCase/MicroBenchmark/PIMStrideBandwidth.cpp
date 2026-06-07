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

#include "PIMStrideBandwidth.h"

#include "gtest/gtest.h"
#include <algorithm>
#include <iomanip>

using namespace DRAMSim;
using namespace std;

PIMStrideBandwidthFixedDataFixture::PIMStrideBandwidthFixedDataFixture()
{
    // Initialize the memory system
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_stride_bw_test", 256 * 16);
    
    max_channels = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, max_channels, 1);

    tCK = getConfigParam(FLOAT, "tCK");
    num_pim_blocks_per_channel = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    scheme = getConfigParam(STRING, "ADDRESS_MAPPING_SCHEME");
    
    // Dynamically read system config parameters to keep address mapping correct
    num_cols = getConfigParam(UINT, "NUM_COLS"); 
    num_rows = getConfigParam(UINT, "NUM_ROWS");
    num_banks = getConfigParam(UINT, "NUM_BANKS");
    cout << "> NUM_COLS: " << num_cols << endl;
    cout << "> NUM_ROWS: " << num_rows << endl;
    cout << "> NUM_BANKS: " << num_banks << endl;

    bytes_per_pim_transaction = num_pim_blocks_per_channel * 32; // 256 Bytes

}

PIMStrideBandwidthFixedDataFixture::~PIMStrideBandwidthFixedDataFixture()
{
    pim_mem_->printStats(true);
}

void PIMStrideBandwidthFixedDataFixture::run()
{
    while (pim_mem_->hasPendingTransactions())
    {
        pim_mem_->update();
    }
}

uint64_t PIMStrideBandwidthFixedDataFixture::getAddressSpaceBytes() const
{
    uint64_t ranks = getConfigParam(UINT, "NUM_RANKS");
    uint64_t bytes_per_column = getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") / 8;

    return static_cast<uint64_t>(max_channels) * ranks * static_cast<uint64_t>(num_banks) *
           static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(num_cols) * bytes_per_column;
}

uint64_t PIMStrideBandwidthFixedDataFixture::makeStrideAddress(uint64_t logical_index,
                                                               uint64_t stride_bytes) const
{
    uint64_t address_space_bytes = getAddressSpaceBytes();
    uint64_t addr = (logical_index * stride_bytes) % address_space_bytes;

    uint64_t transaction_bytes =
        getConfigParam(UINT, "JEDEC_DATA_BUS_BITS") * getConfigParam(UINT, "BL") / 8;

    addr = addr - (addr % transaction_bytes);
    return maskHabBankBits(addr);
}

uint64_t PIMStrideBandwidthFixedDataFixture::maskHabBankBits(uint64_t addr) const
{
    int scheme_id = 8;
    if (scheme.rfind("Scheme", 0) == 0)
    {
        scheme_id = stoi(scheme.substr(6));
    }

    switch (scheme_id)
    {
        case 1:
        case 2:
        case 5:
        case 10:
            return addr & ~(7ULL << 6);
        case 3:
        case 4:
            return addr & ~(7ULL << 25);
        case 6:
            return addr & ~(7ULL << 11);
        case 7:
        case 8:
        case 11:
            return addr & ~(7ULL << 10);
        case 9:
            return addr & ~(7ULL << 15);
        default:
            return addr;
    }
}

void PIMStrideBandwidthFixedDataFixture::printAddressSample(
    const std::vector<uint64_t>& addresses, size_t sample_count) const
{
    cout << "> Address decode sample:" << endl;
    size_t limit = min(sample_count, addresses.size());

    for (size_t i = 0; i < limit; ++i)
    {
        unsigned ch = 0;
        unsigned rank = 0;
        unsigned bank = 0;
        unsigned row = 0;
        unsigned col = 0;

        pim_mem_->addrMapping->addressMapping(addresses[i], ch, rank, bank, row, col);

        cout << "  [" << i << "] addr=0x" << hex << addresses[i] << dec << " ch=" << ch
             << " bank=" << bank << " row=" << row << " col=" << col << endl;
    }
}

void PIMStrideBandwidthFixedDataFixture::testStrideBandwidth(uint64_t stride_bytes,
                                                             uint64_t total_data_size_bytes)
{
    // 1. Calculate the total transaction count
    uint32_t num_transactions =
        (total_data_size_bytes + bytes_per_pim_transaction - 1) / bytes_per_pim_transaction;
    uint64_t actual_data_size = (uint64_t)num_transactions * bytes_per_pim_transaction;

    cout << ">> PIM/HAB Stride Bandwidth Test (Physical Address Mapping)" << endl;
    cout << "> Scheme: " << scheme << endl;
    cout << "> Stride: " << stride_bytes << " Bytes " << endl;
    cout << "> Channels: " << max_channels << endl;
    cout << "> Effective Bytes / Transaction: " << bytes_per_pim_transaction << endl;
    cout << "> Target Data Size: " << total_data_size_bytes / (1024.0 * 1024.0) << " MB" << endl;
    cout << "> Total Transactions: " << num_transactions << endl;

    BurstType null_bst;

    kernel_->setPIMControl(true, 0, false, false);
    kernel_->parkIn();
    run();

    kernel_->changePIMMode(dramMode::SB, dramMode::HAB);
    run();

    std::vector<uint64_t> addresses;
    addresses.reserve(num_transactions);

    for (uint32_t i = 0; i < num_transactions; ++i)
    {
        addresses.push_back(makeStrideAddress(i, stride_bytes));
    }

    // printAddressSample(addresses, 8);
    cout << "> Requests generated. Starting injection..." << endl;

    // 2. Execute the test
    uint64_t cycle = 0;
    uint32_t sent_count = 0;
    uint32_t target_pending = max_channels * 16;
    const uint64_t MAX_CYCLES = 20000000;

    while (sent_count < num_transactions && cycle < MAX_CYCLES)
    {
        uint64_t pending = pim_mem_->hasPendingTransactions();

        if (pending < target_pending)
        {
            uint32_t batch_size = min((uint32_t)(target_pending - pending),
                                      (uint32_t)(num_transactions - sent_count));
            // Increase the injection burst size to better use MC scheduling
            batch_size = min(batch_size, 128u);

            for (uint32_t k = 0; k < batch_size; ++k)
            {
                uint64_t addr = addresses[sent_count];
                if (pim_mem_->addTransaction(false, addr, "STRIDE_MAP", &null_bst))
                {
                    sent_count++;
                }
                else
                {
                    break;
                }
            }
        }

        pim_mem_->update();
        cycle++;
    }

    while (pim_mem_->hasPendingTransactions() && cycle < MAX_CYCLES)
    {
        pim_mem_->update();
        cycle++;
    }

    kernel_->changePIMMode(dramMode::HAB, dramMode::SB);
    kernel_->parkOut();
    run();

    double time_sec = (double)cycle * tCK * 1e-9;
    double bw_gbps = (double)actual_data_size / (1024.0 * 1024.0 * 1024.0) / time_sec;

    cout << "> Results:" << endl;
    cout << "  Stride: " << stride_bytes << " Bytes" << endl;
    cout << "  Data Size: " << actual_data_size / (1024.0 * 1024.0) << " MB" << endl;
    cout << "  Cycles: " << cycle << endl;
    cout << "  Time: " << time_sec * 1000.0 << " ms" << endl;
    cout << "  Bandwidth: " << bw_gbps << " GB/s" << endl;
    cout << "------------------------------------------------" << endl;
}

// ============================================================================
// Test Cases
// ============================================================================

// Fixed data size: 64 MB
const uint64_t FIXED_DATA_SIZE = 64 * 1024 * 1024;

// 1. Stride = 32B (True Sequential, 1 Transaction)
TEST_F(PIMStrideBandwidthFixedDataFixture, stride_bandwidth)
{
    testStrideBandwidth(32, FIXED_DATA_SIZE);
    testStrideBandwidth(64, FIXED_DATA_SIZE);
    testStrideBandwidth(128, FIXED_DATA_SIZE);
    testStrideBandwidth(256, FIXED_DATA_SIZE);
    testStrideBandwidth(512, FIXED_DATA_SIZE);
    testStrideBandwidth(1024, FIXED_DATA_SIZE);
    testStrideBandwidth(2048, FIXED_DATA_SIZE);
    testStrideBandwidth(4096, FIXED_DATA_SIZE);
    testStrideBandwidth(8192, FIXED_DATA_SIZE);
    testStrideBandwidth(16384, FIXED_DATA_SIZE);
    testStrideBandwidth(32768, FIXED_DATA_SIZE);
    testStrideBandwidth(65536, FIXED_DATA_SIZE);
    testStrideBandwidth(131072, FIXED_DATA_SIZE);
}
