/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#ifndef __PIM_REDUCTION_BENCH_H__
#define __PIM_REDUCTION_BENCH_H__

#include "tests/TestCases.h"
#include <memory>
#include <vector>

namespace DRAMSim
{
class PIMReductionBenchFixture : public testing::Test
{
  public:
    PIMReductionBenchFixture();
    virtual ~PIMReductionBenchFixture();

    /**
     * Tree-reduction sum test (multiple passes, stride doubles by burst)
     * Mainly measures the HBM2-PIM cycle cost under strided access
     * @param total_elements: total vector length (number of fp16 elements)
     */
    void testSumReduction(int total_elements);

  protected:
    void run();
    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
};
}
#endif
