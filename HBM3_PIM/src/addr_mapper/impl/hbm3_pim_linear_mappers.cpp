#include <vector>

#include "base/base.h"
#include "dram/dram.h"
#include "addr_mapper/addr_mapper.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class HBM3PIMLinearMapperBase : public IAddrMapper {
  public:
    IDRAM* m_dram = nullptr;

    int m_num_levels = -1;          // How many levels in the hierarchy?
    std::vector<int> m_addr_bits;   // How many address bits for each level in the hierarchy?
    Addr_t m_tx_offset = -1;

    int m_col_bits_idx = -1;
    int m_row_bits_idx = -1;


  protected:
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) {
      m_dram = memory_system->get_ifce<IDRAM>();

      // Populate m_addr_bits vector with the number of address bits for each level in the hierachy
      const auto& count = m_dram->m_organization.count;
      m_num_levels = count.size();
      m_addr_bits.resize(m_num_levels);
      for (size_t level = 0; level < m_addr_bits.size(); level++) {
        m_addr_bits[level] = calc_log2(count[level]);
      }

      // Last (Column) address have the granularity of the prefetch size
      //m_addr_bits[m_num_levels - 1] -= calc_log2(m_dram->m_internal_prefetch_size);

      int tx_bytes = m_dram->m_internal_prefetch_size * m_dram->m_channel_width / 8;
      m_tx_offset = calc_log2(tx_bytes);

      // Determine where are the row and col bits for ChRaBaRoCo and RoBaRaCoCh
      try {
        m_row_bits_idx = m_dram->m_levels("row");
      } catch (const std::out_of_range& r) {
        throw std::runtime_error(fmt::format("Organization \"row\" not found in the spec, cannot use linear mapping!"));
      }

      // Assume column is always the last level
      m_col_bits_idx = m_num_levels - 1;
    }

};


class HBM3PIMMap final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMMap, "HBM3-PIM", "Applies a HBM3-PIM mapping to the address. (CH pCH Ra BG Ba Ro Co)");

  public:
    void init() override { };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }

    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG
      req.addr_vec[2] = slice_lower_bits(addr, m_addr_bits[2]); // Ra
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch
    }
};


class HBM3PIMCustomMap final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMCustomMap, "HBM3-PIM-Custom", "Applies a HBM3-PIM custom mapping to the address. (Pch Ra Ba Ro Co Ch)");

  public:
    void init() override { };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }

    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]);
      for (int i = m_addr_bits.size() - 1; i >= 1; i--) {
        req.addr_vec[i] = slice_lower_bits(addr, m_addr_bits[i]);
      }
    }
};


// Hierarchy index reference (HBM3): 0=Ch  1=Pch  2=Ra  3=BG  4=Ba  5=Ro  6=Co
// In all schemes below, the user-facing "rank" field is mapped to Pch (idx=1, 1 bit),
// and Ra (idx=2) is hard-zeroed (rank dimension unused).
// "bank" in Scheme1-7 names refers to combined BG+BA (4 bits): BA = lower 2 bits, BG = upper 2 bits.


// Scheme1 (MSB->LSB): chan : rank : row : col : bank
// Slice order  LSB->MSB: Ba(2) -> BG(2) -> Co(5) -> Ro(14) -> Pch(1) -> Ch(4)
class HBM3PIMScheme1Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme1Map, "HBM3-PIM-Scheme1", "Scheme1: chan:rank:row:col:bank (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (LSB,   2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[2] = 0;                                       // Ra  unused
    }
};


// Scheme2 (MSB->LSB): chan : row : col : bank : rank  (rank=0bit, Pch hard-zeroed)
// Slice order  LSB->MSB: Ba(2) -> BG(2) -> Co(5) -> Ro(14) -> Ch(4)
class HBM3PIMScheme2Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme2Map, "HBM3-PIM-Scheme2", "Scheme2: chan:row:col:bank:rank (MSB->LSB), rank=0bit");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (LSB,   2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[1] = 0;                                       // Pch (rank=0bit)
      req.addr_vec[2] = 0;                                       // Ra  unused
    }
};


// Scheme3 (MSB->LSB): chan : rank : bank : col : row
// Slice order  LSB->MSB: Ro(14) -> Co(5) -> Ba(2) -> BG(2) -> Pch(1) -> Ch(4)
class HBM3PIMScheme3Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme3Map, "HBM3-PIM-Scheme3", "Scheme3: chan:rank:bank:col:row (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (LSB,  14b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (        2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[2] = 0;
    }
};


// Scheme4 (MSB->LSB): chan : rank : bank : row : col
// Slice order  LSB->MSB: Co(5) -> Ro(14) -> Ba(2) -> BG(2) -> Pch(1) -> Ch(4)
class HBM3PIMScheme4Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme4Map, "HBM3-PIM-Scheme4", "Scheme4: chan:rank:bank:row:col (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (LSB,   5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (        2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[2] = 0;
    }
};


// Scheme5 (MSB->LSB): chan : row : col : rank : bank
// Slice order  LSB->MSB: Ba(2) -> BG(2) -> Pch(1) -> Co(5) -> Ro(14) -> Ch(4)
class HBM3PIMScheme5Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme5Map, "HBM3-PIM-Scheme5", "Scheme5: chan:row:col:rank:bank (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (LSB,   2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[2] = 0;
    }
};


// Scheme6 (MSB->LSB): chan : row : bank : rank : col
// Slice order  LSB->MSB: Co(5) -> Pch(1) -> Ba(2) -> BG(2) -> Ro(14) -> Ch(4)
class HBM3PIMScheme6Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme6Map, "HBM3-PIM-Scheme6", "Scheme6: chan:row:bank:rank:col (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (LSB,   5b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (        2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (MSB,   4b)
      req.addr_vec[2] = 0;
    }
};


// Scheme7 (MSB->LSB): row : col : rank : bank : chan
// Slice order  LSB->MSB: Ch(4) -> Ba(2) -> BG(2) -> Pch(1) -> Co(5) -> Ro(14)
class HBM3PIMScheme7Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme7Map, "HBM3-PIM-Scheme7", "Scheme7: row:col:rank:bank:chan (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (LSB,   4b)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (        2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (        1b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (MSB,  14b)
      req.addr_vec[2] = 0;
    }
};


// Scheme8 (MSB->LSB): rank : row : colH : bg : bank_g : chan
// Slice order  LSB->MSB: Ch(4) -> Ba(2) -> BG(2) -> Co(5) -> Ro(14) -> Pch(1)
class HBM3PIMScheme8Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme8Map, "HBM3-PIM-Scheme8", "Scheme8: rank:row:colH:bg:bank_g:chan (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (LSB,   4b)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (        2b)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG  (        2b)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co  (        5b)
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro  (       14b)
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (MSB,   1b)
      req.addr_vec[2] = 0;                                       // Ra  unused
    }
};


// Scheme9 (MSB->LSB): rank : row : bg : bank_g : colH : chan
// Slice order  LSB->MSB: Ch(4) -> Co(5) -> Ba(2) -> BG(2) -> Ro(14) -> Pch(1)
class HBM3PIMScheme9Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme9Map, "HBM3-PIM-Scheme9", "Scheme9: rank:row:bg:bank_g:colH:chan (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (LSB)
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (MSB)
      req.addr_vec[2] = 0;
    }
};


// Scheme10 (MSB->LSB): rank : row : colH : chan : bg : bank_g
// Slice order  LSB->MSB: Ba(2) -> BG(2) -> Ch(4) -> Co(5) -> Ro(14) -> Pch(1)
class HBM3PIMScheme10Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme10Map, "HBM3-PIM-Scheme10", "Scheme10: rank:row:colH:chan:bg:bank_g (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba  (LSB)
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (MSB)
      req.addr_vec[2] = 0;
    }
};


// Scheme11 (MSB->LSB): rank : colH : row : bg : bank_g : chan
// Slice order  LSB->MSB: Ch(4) -> Ba(2) -> BG(2) -> Ro(14) -> Co(5) -> Pch(1)
class HBM3PIMScheme11Map final : public HBM3PIMLinearMapperBase, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, HBM3PIMScheme11Map, "HBM3-PIM-Scheme11", "Scheme11: rank:colH:row:bg:bank_g:chan (MSB->LSB)");

  public:
    void init() override { };
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      HBM3PIMLinearMapperBase::setup(frontend, memory_system);
    }
    void apply(Request& req) override {
      req.addr_vec.resize(m_num_levels, -1);
      Addr_t addr = req.addr >> m_tx_offset;
      req.addr_vec[0] = slice_lower_bits(addr, m_addr_bits[0]); // Ch  (LSB)
      req.addr_vec[4] = slice_lower_bits(addr, m_addr_bits[4]); // Ba
      req.addr_vec[3] = slice_lower_bits(addr, m_addr_bits[3]); // BG
      req.addr_vec[5] = slice_lower_bits(addr, m_addr_bits[5]); // Ro
      req.addr_vec[6] = slice_lower_bits(addr, m_addr_bits[6]); // Co
      req.addr_vec[1] = slice_lower_bits(addr, m_addr_bits[1]); // Pch (MSB)
      req.addr_vec[2] = 0;
    }
};


}   // namespace Ramulator
