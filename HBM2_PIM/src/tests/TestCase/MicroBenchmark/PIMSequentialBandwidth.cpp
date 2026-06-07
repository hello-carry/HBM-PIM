#include "PIMSequentialBandwidth.h"

#include "gtest/gtest.h"

/*
 * PIM Sequential Bandwidth Tests
 *
 * Fully sequential access test (crosses channels and rows according to the
 * current system channel count)
 * - The control variable is access data size
 * - Addresses are generated in Scheme8 channel-interleaved order
 * - Start with small data sizes and gradually increase them to observe the
 *   effects of row and channel crossing
 *
 * Data size versus row/channel crossing:
 * - 256 B (1 transaction): no row crossing, single channel
 * - 512 B (2 transactions): no row crossing, 2 channels
 * - 1 KB (4 transactions): no row crossing, 4 channels
 * - 2 KB (8 transactions): no row crossing, 8 channels
 * - 4 KB (16 transactions): use all current 16 channels
 * - 8 KB+: increase sequential column accesses on all available channels,
 *   crossing rows when necessary
 */

using namespace DRAMSim;

TEST_F(PIMSequentialBandwidthFixture, sequential_bandwidth)
{
    testBandwidthByDataSize(1048576, 512); 

}

