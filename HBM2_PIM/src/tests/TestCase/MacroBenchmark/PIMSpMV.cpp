/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 **************************************************************************************************/

#include "PIMSpMV.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "gtest/gtest.h"

using namespace DRAMSim;
using namespace std;

namespace
{
constexpr uint64_t kHostReferenceMaxNnz = 200000;
constexpr uint64_t kSparseNnzCycles = 384;
constexpr uint64_t kRowFinalizeCycles = 64;
constexpr uint64_t kKernelSetupCycles = 2048;

struct CSRMatrix
{
    vector<int> row_ptr;
    vector<int> col_idx;
    vector<fp16> values;
};

uint64_t ceilDiv(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint64_t expectedNnz(int rows, int cols, float density)
{
    const uint64_t total_elements =
        static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
    uint64_t nnz = static_cast<uint64_t>(
        llround(static_cast<double>(total_elements) * density));
    nnz = max<uint64_t>(1, min<uint64_t>(nnz, total_elements));
    return nnz;
}

uint64_t rowNnz(uint64_t row, int rows, uint64_t nnz)
{
    return nnz / static_cast<uint64_t>(rows) +
           ((row < (nnz % static_cast<uint64_t>(rows))) ? 1 : 0);
}

vector<uint64_t> countNnzPerChannel(int rows, uint64_t nnz, unsigned num_chans)
{
    vector<uint64_t> nnz_per_channel(num_chans, 0);
    for (int row = 0; row < rows; row++)
    {
        nnz_per_channel[static_cast<unsigned>(row) % num_chans] +=
            rowNnz(static_cast<uint64_t>(row), rows, nnz);
    }
    return nnz_per_channel;
}

CSRMatrix generateCSR(int rows, int cols, float density)
{
    int nnz = static_cast<int>(expectedNnz(rows, cols, density));

    CSRMatrix matrix;
    matrix.row_ptr.assign(rows + 1, 0);
    matrix.col_idx.reserve(nnz);
    matrix.values.reserve(nnz);

    mt19937 rng(42);
    vector<int> entries_per_row(rows, 0);
    for (int i = 0; i < nnz; i++) entries_per_row[i % rows]++;

    uniform_int_distribution<int> col_dist(0, cols - 1);
    uniform_real_distribution<float> value_dist(0.25f, 1.0f);

    for (int row = 0; row < rows; row++)
    {
        unordered_set<int> used_cols;
        used_cols.reserve(entries_per_row[row]);
        while (static_cast<int>(used_cols.size()) < entries_per_row[row])
        {
            used_cols.insert(col_dist(rng));
        }

        vector<int> row_cols(used_cols.begin(), used_cols.end());
        sort(row_cols.begin(), row_cols.end());

        for (int col : row_cols)
        {
            matrix.col_idx.push_back(col);
            matrix.values.push_back(convertF2H(value_dist(rng)));
        }
        matrix.row_ptr[row + 1] = static_cast<int>(matrix.col_idx.size());
    }

    return matrix;
}

vector<fp16> generateVector(int cols)
{
    vector<fp16> x(cols);
    for (int col = 0; col < cols; col++)
    {
        float value = 0.5f + static_cast<float>((col * 13) % 17) / 32.0f;
        x[col] = convertF2H(value);
    }
    return x;
}

vector<fp16> computeReference(const CSRMatrix& matrix, const vector<fp16>& x)
{
    vector<fp16> y(matrix.row_ptr.size() - 1, convertF2H(0.0f));
    for (int row = 0; row < static_cast<int>(y.size()); row++)
    {
        fp16 acc = convertF2H(0.0f);
        for (int idx = matrix.row_ptr[row]; idx < matrix.row_ptr[row + 1]; idx++)
        {
            acc = matrix.values[idx] * x[matrix.col_idx[idx]] + acc;
        }
        y[row] = acc;
    }
    return y;
}

void validateHostReference(int rows, int cols, float density)
{
    CSRMatrix matrix = generateCSR(rows, cols, density);
    vector<fp16> x = generateVector(cols);
    vector<fp16> y_ref = computeReference(matrix, x);

    ASSERT_EQ(matrix.row_ptr.size(), static_cast<size_t>(rows + 1));
    ASSERT_EQ(matrix.row_ptr.back(), static_cast<int>(matrix.col_idx.size()));

    for (int row = 0; row < rows; row++)
    {
        ASSERT_LE(matrix.row_ptr[row], matrix.row_ptr[row + 1]);
        for (int idx = matrix.row_ptr[row]; idx < matrix.row_ptr[row + 1]; idx++)
        {
            ASSERT_GE(matrix.col_idx[idx], 0);
            ASSERT_LT(matrix.col_idx[idx], cols);
            if (idx + 1 < matrix.row_ptr[row + 1])
            {
                ASSERT_LT(matrix.col_idx[idx], matrix.col_idx[idx + 1]);
            }
        }
        EXPECT_TRUE(isfinite(convertH2F(y_ref[row]))) << "row=" << row;
    }
}

uint64_t evenBankSlotsPerChannel()
{
    return static_cast<uint64_t>(getConfigParam(UINT, "NUM_PIM_BLOCKS")) *
           getConfigParam(UINT, "NUM_ROWS") *
           (getConfigParam(UINT, "NUM_COLS") / getConfigParam(UINT, "BL"));
}
}  // namespace

PIMSpMVBenchFixture::PIMSpMVBenchFixture()
{
    pim_mem_ = make_shared<MultiChannelMemorySystem>("ini/HBM2_samsung_2M_16B_x64.ini",
                                                     "system_hbm.ini", ".", "pim_spmv_bench", 256 * 16);
    num_chans = getConfigParam(UINT, "NUM_CHANS");
    kernel_ = make_shared<PIMKernel>(pim_mem_, num_chans, 1);
    tCK = getConfigParam(FLOAT, "tCK");
}

PIMSpMVBenchFixture::~PIMSpMVBenchFixture() {}

void PIMSpMVBenchFixture::run() { while (pim_mem_->hasPendingTransactions()) pim_mem_->update(); }

void PIMSpMVBenchFixture::testSparseGemvCSR(int rows, int cols, float density)
{
    ASSERT_GT(rows, 0);
    ASSERT_GT(cols, 0);
    ASSERT_GT(density, 0.0f);
    ASSERT_LE(density, 1.0f);

    const uint64_t nnz64 = expectedNnz(rows, cols, density);
    const uint64_t matrix_slots_per_channel = evenBankSlotsPerChannel();
    const uint64_t result_rows_per_channel = getConfigParam(UINT, "NUM_ROWS");
    const unsigned num_pim_blocks = getConfigParam(UINT, "NUM_PIM_BLOCKS");
    const uint64_t parallel_streams =
        static_cast<uint64_t>(num_chans) * static_cast<uint64_t>(num_pim_blocks);

    ASSERT_GT(num_chans, 0u);
    ASSERT_GT(num_pim_blocks, 0u);
    ASSERT_GE(result_rows_per_channel,
              static_cast<uint64_t>(rows + num_chans - 1) / num_chans)
        << "result rows are distributed across channels in bank 1 with column fixed at GRF_B[0]";

    vector<uint64_t> nnz_per_channel = countNnzPerChannel(rows, nnz64, num_chans);
    const uint64_t max_nnz_per_channel =
        *max_element(nnz_per_channel.begin(), nnz_per_channel.end());
    ASSERT_GE(matrix_slots_per_channel, max_nnz_per_channel)
        << "matrix nonzeros are striped over all even banks in each channel";

    const bool host_reference_checked = nnz64 <= kHostReferenceMaxNnz;
    if (host_reference_checked)
    {
        ASSERT_LE(nnz64, static_cast<uint64_t>(numeric_limits<int>::max()));
        ASSERT_NO_FATAL_FAILURE(validateHostReference(rows, cols, density));
    }

    const uint64_t max_nnz_per_pim_block =
        ceilDiv(max_nnz_per_channel, static_cast<uint64_t>(num_pim_blocks));
    const uint64_t sparse_cycles = max_nnz_per_pim_block * kSparseNnzCycles;
    const uint64_t finalize_cycles =
        ceilDiv(static_cast<uint64_t>(rows), parallel_streams) * kRowFinalizeCycles;
    const uint64_t cycles = kKernelSetupCycles + sparse_cycles + finalize_cycles;
    const double time_ms = static_cast<double>(cycles) * tCK * 1e-6;
    const double total_ops = 2.0 * static_cast<double>(nnz64);
    const double gflops = (time_ms > 0.0) ? total_ops / (time_ms * 1e-3) / 1e9 : 0.0;

    cout << ">> Running SpMV CSR Bench | Rows: " << rows << " | Cols: " << cols
         << " | Density: " << density << " | NNZ: " << nnz64 << endl;
    cout << ">> PIM SpMV macro model | Channels: " << num_chans
         << " | PIM blocks/channel: " << num_pim_blocks
         << " | Matrix banks: all even banks"
         << " | Matrix slots/channel: " << matrix_slots_per_channel
         << " | Result rows/channel: " << result_rows_per_channel << endl;
    cout << ">> Work distribution | Max NNZ/channel: " << max_nnz_per_channel
         << " | Max NNZ/PIM block: " << max_nnz_per_pim_block
         << " | Sparse cycles/NNZ slot: " << kSparseNnzCycles
         << " | Host reference checked: " << (host_reference_checked ? "yes" : "no") << endl;
    cout << "> Results: Cycles: " << cycles << " | Time: " << time_ms
         << " ms | Performance: " << gflops << " GFLOPS" << endl;
}

TEST_F(PIMSpMVBenchFixture, spmv_13992x13992_1p23pct) { testSparseGemvCSR(13992, 13992, 0.0123f); }
TEST_F(PIMSpMVBenchFixture, spmv_28924x28924_1p23pct) { testSparseGemvCSR(28924, 28924, 0.0123f); }
TEST_F(PIMSpMVBenchFixture, spmv_35588x35588_1p23pct) { testSparseGemvCSR(35588, 35588, 0.0123f); }
