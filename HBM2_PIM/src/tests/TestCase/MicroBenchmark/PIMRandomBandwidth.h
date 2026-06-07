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

#ifndef __PIM_RANDOM_BANDWIDTH_TEST_CASE_H__
#define __PIM_RANDOM_BANDWIDTH_TEST_CASE_H__

#include <memory>
#include <string>
#include <vector>
#include <random>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMRandomBandwidthFixture: true random memory-access bandwidth test
 *
 * Goals:
 * - Validate the worst-case/actual bandwidth under random access in HAB mode.
 * - Strictly follow the HAB-mode hardware constraint: Bank can only be 0 or 1
 *   (Even/Odd Ping-Pong).
 * - Avoid the reserved row region (Row < 8192).
 *
 * Randomness strategy (complete randomness):
 * 1. Random Channel: randomly distribute across all available channels
 *    (or round-robin to keep the system saturated).
 * 2. Random Row: randomly choose from [0, 8191] to force row-buffer misses.
 * 3. Random Bank: randomly choose from {0, 1} to test even/odd bank concurrency
 *    and switching.
 * 4. Random Column: randomly choose from the legal column range.
 *
 * Differences from older tests:
 * - Old ScalableBandwidth: sequential access, uses only Bank 0, mainly row hits.
 * - Old RandomTestCases: may not strictly limit Bank 0/1, or may use different
 *   randomness granularity.
 * - This test: fully random at transaction granularity, tuned specifically for
 *   HAB constraints.
 */

namespace DRAMSim
{
class PIMRandomBandwidthFixture : public testing::Test
{
  public:
    PIMRandomBandwidthFixture();
    virtual ~PIMRandomBandwidthFixture();

    // Core test function
    // @param data_size_bytes: total data size to test, in bytes
    // @param num_channels: number of channels to use (default 64)
    void testRandomBandwidthByDataSize(uint64_t data_size_bytes, uint32_t num_channels = 64);

  protected:
    void run();

    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned max_channels;
    unsigned num_pim_blocks_per_channel;
    unsigned bytes_per_pim_transaction; // 256 bytes
    unsigned num_cols;
    unsigned num_rows;
    unsigned num_ranks;
    unsigned num_banks; // Number of physical banks
    uint32_t pim_reg_ra_threshold_; // 8192

    std::mt19937 rng_;
};

}  // namespace DRAMSim

#endif /*__PIM_RANDOM_BANDWIDTH_TEST_CASE_H__*/
