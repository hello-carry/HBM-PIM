/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#ifndef __PIM_SPMV_BENCH_H__
#define __PIM_SPMV_BENCH_H__

#include "tests/TestCases.h"
#include <memory>
#include <random>

namespace DRAMSim
{
class PIMSpMVBenchFixture : public testing::Test
{
  public:
    PIMSpMVBenchFixture();
    virtual ~PIMSpMVBenchFixture();

    void testSparseGemvCSR(int rows, int cols, float density);

  protected:
    void run();
    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
};
}
#endif
