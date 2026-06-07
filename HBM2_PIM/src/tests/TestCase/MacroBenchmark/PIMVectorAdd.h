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

#ifndef __PIM_VECTOR_ADD_MACRO_BENCH_H__
#define __PIM_VECTOR_ADD_MACRO_BENCH_H__

#include <memory>
#include <string>
#include <vector>

#include "tests/TestCases.h"

using namespace DRAMSim;

/*
 * PIMVectorAddFixture: macro benchmark - vector add
 *
 * Logic:
 * 1. Control variable: vector length N (num_elements).
 * 2. Storage: Vector A (Row 0), Vector B (Row 100), Result (Row 200).
 * 3. Flow (driven by PIMKernel::executeEltwise + computeAddOrMul):
 *    - For each even/odd bank:
 *      Read Bank (Vec A) -> GRF (FILL)
 *      Read Bank (Vec B) + GRF -> GRF (ADD)
 *      Write back to Bank (Result) (GRF_TO_BANK)
 */

namespace DRAMSim
{
class PIMVectorAddFixture : public testing::Test
{
  public:
    PIMVectorAddFixture();
    virtual ~PIMVectorAddFixture();

    // Core test function
    // @param num_elements: number of elements in the vector (FP16/INT8/FP32 depends on the config)
    void testVectorAdd(uint64_t num_elements);

  protected:
    void run();
    uint64_t measureNonPIMCycles(uint64_t num_elements);

    shared_ptr<MultiChannelMemorySystem> pim_mem_;
    shared_ptr<PIMKernel> kernel_;
    float tCK;
    unsigned num_chans;
    PIMPrecision precision;
};

}  // namespace DRAMSim

#endif /*__PIM_VECTOR_ADD_MACRO_BENCH_H__*/
