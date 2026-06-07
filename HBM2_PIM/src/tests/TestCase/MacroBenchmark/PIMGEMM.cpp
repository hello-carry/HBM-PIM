#include "PIMGEMM.h"

#include "gtest/gtest.h"

using namespace DRAMSim;

TEST_F(PIMGEMMBenchFixture, gemm_2K_1K_2K)
{
    setPIMGEMMBenchTestCase(2048, 1024, 2048);  // (M, K, N): A=2048x1024, B=1024x2048
    executePIMKernel();
    // executeKernel();
}

TEST_F(PIMGEMMBenchFixture, gemm_1K_1K_1K)
{
    // setPIMGEMMBenchTestCase(10, 10, 10);  // A=1024x1024, B=1024x1024
    // setPIMGEMMBenchTestCase(100, 100, 100);  // A=1024x1024, B=1024x1024
    setPIMGEMMBenchTestCase(1024, 1024, 1024);  // A=1024x1024, B=1024x1024
    executePIMKernel();
    // executeKernel();
}

TEST_F(PIMGEMMBenchFixture, gemm_1K_2K_1K)
{
    setPIMGEMMBenchTestCase(1024, 2048, 1024);  // A=1024x2048, B=2048x1024
    executePIMKernel();
    // executeKernel();
}
