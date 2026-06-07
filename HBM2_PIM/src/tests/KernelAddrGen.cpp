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

#include <iomanip>

#include "AddressMapping.h"
#include "tests/PIMKernel.h"

unsigned PIMAddrManager::maskByBit(unsigned value, int start, int end)
{
    int length = start - end + 1;
    value = value >> end;
    return value & ((1 << length) - 1);
}

uint64_t PIMAddrManager::addrGen(unsigned chan, unsigned rank, unsigned bankgroup, unsigned bank,
                                 unsigned row, unsigned col)
{
    /**
     * HBM2 16-Channel 物理地址位宽划分参考:
     * [31-18] Row      : 14 bits
     * [17-13] ColHigh  : 5 bits
     * [12-9]  BankFlat : 4 bits (BankGroup + Bank)
     * [8-5]   Channel  : 4 bits
     * [4-0]   Offset   : 5 bits (32B Aligned)
     */
    uint64_t addr = 0;
    unsigned flat_bank = (bankgroup << num_bank_bits_) | bank;
    unsigned total_bank_bits = num_bank_bits_ + num_bankgroup_bits_;

    if (address_mapping_scheme_ == Scheme1)
    {
        // Scheme1: [MSB] chan(4) : rank(0) : row(14) : col(5) : bg(2) : bank(2) : offset(5) [LSB]
        addr = chan;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme2)
    {
        // Scheme2: [MSB] chan(4) : row(14) : col(5) : bg(2) : bank(2) : rank(0) : offset(5) [LSB]
        addr = chan;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme3)
    {
        // Scheme3: [MSB] chan(4) : rank(0) : bg(2) : bank(2) : col(5) : row(14) : offset(5) [LSB]
        addr = chan;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme4)
    {
        // Scheme4: [MSB] chan(4) : rank(0) : bg(2) : bank(2) : row(14) : col(5) : offset(5) [LSB]
        addr = chan;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme5)
    {
        // Scheme5: [MSB] chan(4) : row(14) : col(5) : rank(0) : bg(2) : bank(2) : offset(5) [LSB]
        addr = chan;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme6)
    {
        // Scheme6: [MSB] chan(4) : row(14) : bg(2) : bank(2) : rank(0) : col(5) : offset(5) [LSB]
        addr = chan;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme7)
    {
        // Scheme7: [MSB] row(14) : col(5) : rank(0) : bg(2) : bank(2) : chan(4) : offset(5) [LSB]
        addr = row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_rank_bits_;    addr |= rank;
        addr <<= total_bank_bits;   addr |= flat_bank;
        addr <<= num_chan_bits_;    addr |= chan;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme8)
    {
        // Scheme8: [MSB] rank(0) : row(14) : col(5) : bg(2) : bank(2) : chan(4) : offset(5) [LSB]
        addr = rank;

        addr <<= num_row_bits_;
        addr |= row;

        addr <<= num_col_bits_;
        addr |= col;

        addr <<= num_bankgroup_bits_;
        addr |= bankgroup;

        addr <<= num_bank_bits_;
        addr |= bank;

        addr <<= num_chan_bits_;
        addr |= chan;

        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme9)
    {
        // Scheme9: [MSB] rank(0) : row(14) : bg(2) : bank(2) : col(5) : chan(4) : offset(5) [LSB]
        // Scheme10: [MSB] rank(0) : row(14) : col(5) : chan(4) : bg(2) : bank(2) : offset(5) [LSB]
        // Scheme11: [MSB] rank(0) : col(5) : row(14) : bg(2) : bank(2) : chan(4) : offset(5) [LSB]
        // 设计：通道优先，然后是列，最后是 BankGroup。
        addr = rank;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_bankgroup_bits_; addr |= bankgroup;
        addr <<= num_bank_bits_;    addr |= bank;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_chan_bits_;    addr |= chan;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme10)
    {
        // Scheme10: [MSB] rank(0) : row(14) : col(5) : chan(4) : bg(2) : bank(2) : offset(5) [LSB]
        // 设计：BankGroup 优先，直接触发物理阵列并行。
        addr = rank;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_chan_bits_;    addr |= chan;
        addr <<= num_bankgroup_bits_; addr |= bankgroup;
        addr <<= num_bank_bits_;    addr |= bank;
        addr <<= num_offset_bits_;
    }
    else if (address_mapping_scheme_ == Scheme11)
    {
        // Scheme11: [MSB] rank(0) : col(5) : row(14) : bg(2) : bank(2) : chan(4) : offset(5) [LSB]
        // 设计：将行放在中间，测试高并行的通道交织。
        addr = rank;
        addr <<= num_col_bits_;     addr |= col;
        addr <<= num_row_bits_;     addr |= row;
        addr <<= num_bankgroup_bits_; addr |= bankgroup;
        addr <<= num_bank_bits_;    addr |= bank;
        addr <<= num_chan_bits_;    addr |= chan;
        addr <<= num_offset_bits_;
    }
    else
    {
        cerr << "Fatal: Not supported address scheme for PIM controller" << endl;
    }
    return addr;
}

uint64_t PIMAddrManager::addrGenSafe(unsigned chan, unsigned rank, unsigned bankgroup,
                                     unsigned bank, unsigned& row, unsigned& col)
{
    while (col >= num_cols_per_bl_)
    {
        row++;
        col -= (num_cols_per_bl_);
    }

    if (row >= num_rows_)
    {
        cerr << "row overflow" << endl;
    }
    return addrGen(chan, rank, bankgroup, bank, row, col);
}
