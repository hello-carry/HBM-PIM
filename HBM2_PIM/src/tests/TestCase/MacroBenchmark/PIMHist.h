/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#ifndef __PIM_HIST_BENCH_H__
#define __PIM_HIST_BENCH_H__

#include "tests/TestCases.h"
#include <memory>

namespace DRAMSim
{
class PIMHistBenchFixture : public testing::Test
{
  public:
    PIMHistBenchFixture();
    virtual ~PIMHistBenchFixture();

    void testHist(uint64_t num_elements, uint32_t num_bins);

  protected:
    void run();
    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
};
}
#endif
