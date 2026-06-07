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

#include "PIMGEMV.h"

#include "gtest/gtest.h"

using namespace DRAMSim;
TEST_F(PIMBenchFixture, gemv_1K_8K)
{
    setPIMBenchTestCase(KernelType::GEMV, 1 * 1024, 8 * 1024);  // (KernelType, out_vec, in_vec)
    executePIMKernel();                                 // execute w/ PIM
    executeKernel();                                    // execute w/o PIM
    expectPIMBench(2.0);
}
TEST_F(PIMBenchFixture, gemv_1K_1K)
{
    setPIMBenchTestCase(KernelType::GEMV, 1 * 1024, 1 * 1024);  // (KernelType, out_vec, in_vec)
    executePIMKernel();                                 // execute w/ PIM
    executeKernel();                                    // execute w/o PIM
    expectPIMBench(2.0);
}

TEST_F(PIMBenchFixture, gemv_8K_1K)
{
    setPIMBenchTestCase(KernelType::GEMV, 8 * 1024, 1 * 1024);  // (KernelType, out_vec, in_vec)
    executePIMKernel();                                 // execute w/ PIM
    executeKernel();                                    // execute w/o PIM
    expectPIMBench(2.0);
}

// TEST_F(PIMBenchFixture, gemv_8K_4K)
// {
//     setPIMBenchTestCase(KernelType::GEMV, 8 * 1024, 4 * 1024);  // (KernelType, out_vec, in_vec)
//     executePIMKernel();                                 // execute w/ PIM
//     executeKernel();                                    // execute w/o PIM
//     expectPIMBench(2.0);
// }
