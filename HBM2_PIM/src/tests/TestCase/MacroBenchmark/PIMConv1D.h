/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#ifndef __PIM_CONV1D_BENCH_H__
#define __PIM_CONV1D_BENCH_H__

#include "tests/TestCases.h"
#include <memory>

namespace DRAMSim
{
class PIMConv1DBenchFixture : public testing::Test
{
  public:
    PIMConv1DBenchFixture();
    virtual ~PIMConv1DBenchFixture();

    void testConv1D(int length);

  protected:
    void run();
    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
};
}
#endif
