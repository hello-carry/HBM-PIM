#ifndef __PIM_GEMM_BENCH_TEST_CASE_H__
#define __PIM_GEMM_BENCH_TEST_CASE_H__

#include <memory>
#include <string>

#include "PIMGEMV.h"

using namespace DRAMSim;

class GemmPIMBenchTest : public PIMBenchTestCase
{
  public:
    GemmPIMBenchTest(unsigned M, unsigned K, unsigned N)
        : PIMBenchTestCase(KernelType::GEMV, /*batch=*/N, /*out=*/M, /*in=*/K), M_(M), K_(K), N_(N)
    {
    }

    uint64_t measureCycle(bool is_pim) override
    {
        uint64_t cycle = 0;
        uint64_t starting_addr = 0;

        if (is_pim)
        {
            kernel_->executeGemv(&dim_data_->weight_npbst_, &dim_data_->input_npbst_,
                                 /*is_tree=*/false);
            kernel_->runPIM();
            cycle = kernel_->getCycle();
        }
        else
        {
            // getDataSize(1) == precision_bytes (1*1*1*precision); avoids private getPrecisionToByte
            uint32_t precision_bytes = dim_data_->getDataSize(1);
            uint32_t a_bytes = M_ * K_ * precision_bytes;
            uint32_t b_bytes = K_ * N_ * precision_bytes;
            uint32_t c_bytes = M_ * N_ * precision_bytes;
            starting_addr = genMemTraffic(mem_, false, a_bytes, starting_addr);
            starting_addr = genMemTraffic(mem_, false, b_bytes, starting_addr);
            run(mem_, &cycle);
            genMemTraffic(mem_, true, c_bytes, starting_addr);
            run(mem_, &cycle);
        }
        return cycle;
    }

    uint64_t getTotalDataSizeGemm() const
    {
        uint32_t precision_bytes = dim_data_->getDataSize(1);
        return (uint64_t)(M_ * K_ + K_ * N_ + M_ * N_) * precision_bytes;
    }

    unsigned getM() const { return M_; }
    unsigned getK() const { return K_; }
    unsigned getN() const { return N_; }

  private:
    unsigned M_, K_, N_;
};

class PIMGEMMBenchFixture : public testing::Test
{
  public:
    virtual void SetUp()
    {
        pim_cycle_ = 0;
        non_pim_cycle_ = 0;
        perfTest = nullptr;
        cout << ">>GEMM Performance Test" << endl;
    }


    void setPIMGEMMBenchTestCase(unsigned M, unsigned K, unsigned N)
    {
        perfTest = new GemmPIMBenchTest(M, K, N);
    }

    void executePIMKernel()
    {
        cout << "  GEMM (PIM enabled) | M=" << perfTest->getM() << " K=" << perfTest->getK()
             << " N=" << perfTest->getN() << endl;
        cout << "  Channels used : " << getConfigParam(UINT, "NUM_CHANS") << endl;
        pim_cycle_ = perfTest->measureCycle(true);
        printStats(pim_cycle_);
    }

    void executeKernel()
    {
        cout << "  GEMM (PIM disabled) | M=" << perfTest->getM() << " K=" << perfTest->getK()
             << " N=" << perfTest->getN() << endl;
        cout << "  Channels used : " << getConfigParam(UINT, "NUM_CHANS") << endl;
        non_pim_cycle_ = perfTest->measureCycle(false);
        printStats(non_pim_cycle_);
    }

    void expectPIMBench(float expected_perf_gain)
    {
        EXPECT_TRUE((float)non_pim_cycle_ / pim_cycle_ > expected_perf_gain)
            << (float)non_pim_cycle_ / pim_cycle_ << endl;
    }

    void printStats(uint64_t cycle)
    {
        cout << "> Test Results " << endl;
        cout << "> Cycle : " << cycle << endl;

        float tCK = getConfigParam(FLOAT, "tCK");
        double time_sec = (double)cycle * tCK * 1e-9;
        cout << "tCK : " << tCK << " ns" << endl;
        cout << "> Time : " << (time_sec * 1000.0) << " ms" << endl;
        cout << endl;
    }

  private:
    uint64_t pim_cycle_, non_pim_cycle_;
    GemmPIMBenchTest *perfTest;
};

#endif /* __PIM_GEMM_BENCH_TEST_CASE_H__ */
