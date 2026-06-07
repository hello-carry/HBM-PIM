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

#ifndef __PIM_STRIDE_BANDWIDTH_FIXED_DATA_TEST_CASE_H__
#define __PIM_STRIDE_BANDWIDTH_FIXED_DATA_TEST_CASE_H__

#include <memory>
#include <string>
#include <vector>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMStrideBandwidthFixedDataFixture: fixed physical-address stride benchmark.
 *
 * Goal:
 * - Feed the same physical address stream into every ADDRESS_MAPPING_SCHEME.
 * - Let each scheme decode that stream into different channel/bank/row/column patterns.
 * - Measure the resulting bandwidth difference while keeping the existing effective
 *   PIM-byte accounting used by this benchmark.
 */

namespace DRAMSim
{
class PIMStrideBandwidthFixedDataFixture : public testing::Test
{
  public:
    PIMStrideBandwidthFixedDataFixture();
    virtual ~PIMStrideBandwidthFixedDataFixture();

    // Core test function
    // @param stride_bytes: address increment after each access, in bytes
    // @param total_data_size_bytes: total test data size, in bytes
    void testStrideBandwidth(uint64_t stride_bytes, uint64_t total_data_size_bytes);

  protected:
    void run();
    uint64_t getAddressSpaceBytes() const;
    uint64_t makeStrideAddress(uint64_t logical_index, uint64_t stride_bytes) const;
    uint64_t maskHabBankBits(uint64_t addr) const;
    void printAddressSample(const std::vector<uint64_t>& addresses, size_t sample_count) const;

    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned max_channels;
    unsigned num_pim_blocks_per_channel;
    unsigned bytes_per_pim_transaction; // 256 bytes
    unsigned num_banks;
    unsigned num_rows;
    unsigned num_cols; // Burst-aligned columns
    std::string scheme;
};

}  // namespace DRAMSim

#endif /*__PIM_STRIDE_BANDWIDTH_FIXED_DATA_TEST_CASE_H__*/
