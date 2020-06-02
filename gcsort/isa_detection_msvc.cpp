#include "isa_detection.h"

#if _MSC_VER
#include <vector>
#include <bitset>
#include <array>
#include <string>
#include <intrin.h>

namespace gcsort {
class InstructionSet {
  // forward declarations
  class InstructionSet_Internal;

 public:
  // getters
  static std::string Vendor(void) { return CPU_Rep._vendor; }
  static std::string Brand(void) { return CPU_Rep._brand; }

  static bool SSE3(void) { return CPU_Rep._func_1_ECX[0]; }
  static bool PCLMULQDQ(void) { return CPU_Rep._func_1_ECX[1]; }
  static bool MONITOR(void) { return CPU_Rep._func_1_ECX[3]; }
  static bool SSSE3(void) { return CPU_Rep._func_1_ECX[9]; }
  static bool FMA(void) { return CPU_Rep._func_1_ECX[12]; }
  static bool CMPXCHG16B(void) { return CPU_Rep._func_1_ECX[13]; }
  static bool SSE41(void) { return CPU_Rep._func_1_ECX[19]; }
  static bool SSE42(void) { return CPU_Rep._func_1_ECX[20]; }
  static bool MOVBE(void) { return CPU_Rep._func_1_ECX[22]; }
  static bool POPCNT(void) { return CPU_Rep._func_1_ECX[23]; }
  static bool AES(void) { return CPU_Rep._func_1_ECX[25]; }
  static bool XSAVE(void) { return CPU_Rep._func_1_ECX[26]; }
  static bool OSXSAVE(void) { return CPU_Rep._func_1_ECX[27]; }
  static bool AVX(void) { return CPU_Rep._func_1_ECX[28]; }
  static bool F16C(void) { return CPU_Rep._func_1_ECX[29]; }
  static bool RDRAND(void) { return CPU_Rep._func_1_ECX[30]; }
  static bool MSR(void) { return CPU_Rep._func_1_EDX[5]; }
  static bool CX8(void) { return CPU_Rep._func_1_EDX[8]; }
  static bool SEP(void) { return CPU_Rep._func_1_EDX[11]; }
  static bool CMOV(void) { return CPU_Rep._func_1_EDX[15]; }
  static bool CLFSH(void) { return CPU_Rep._func_1_EDX[19]; }
  static bool MMX(void) { return CPU_Rep._func_1_EDX[23]; }
  static bool FXSR(void) { return CPU_Rep._func_1_EDX[24]; }
  static bool SSE(void) { return CPU_Rep._func_1_EDX[25]; }
  static bool SSE2(void) { return CPU_Rep._func_1_EDX[26]; }
  static bool FSGSBASE(void) { return CPU_Rep._func_7_EBX[0]; }
  static bool BMI1(void) { return CPU_Rep._func_7_EBX[3]; }
  static bool HLE(void) { return CPU_Rep._isIntel && CPU_Rep._func_7_EBX[4]; }
  static bool AVX2(void) { return CPU_Rep._func_7_EBX[5]; }
  static bool BMI2(void) { return CPU_Rep._func_7_EBX[8]; }
  static bool ERMS(void) { return CPU_Rep._func_7_EBX[9]; }
  static bool INVPCID(void) { return CPU_Rep._func_7_EBX[10]; }
  static bool RTM(void) { return CPU_Rep._isIntel && CPU_Rep._func_7_EBX[11]; }
  static bool AVX512F(void) { return CPU_Rep._func_7_EBX[16]; }
  static bool RDSEED(void) { return CPU_Rep._func_7_EBX[18]; }
  static bool ADX(void) { return CPU_Rep._func_7_EBX[19]; }
  static bool AVX512PF(void) { return CPU_Rep._func_7_EBX[26]; }
  static bool AVX512ER(void) { return CPU_Rep._func_7_EBX[27]; }
  static bool AVX512CD(void) { return CPU_Rep._func_7_EBX[28]; }
  static bool SHA(void) { return CPU_Rep._func_7_EBX[29]; }

  static bool PREFETCHWT1(void) { return CPU_Rep._func_7_ECX[0]; }

  static bool LAHF(void) { return CPU_Rep._func_81_ECX[0]; }
  static bool LZCNT(void) {
    return CPU_Rep._isIntel && CPU_Rep._func_81_ECX[5];
  }
  static bool ABM(void) { return CPU_Rep._isAMD && CPU_Rep._func_81_ECX[5]; }
  static bool SSE4a(void) { return CPU_Rep._isAMD && CPU_Rep._func_81_ECX[6]; }
  static bool XOP(void) { return CPU_Rep._isAMD && CPU_Rep._func_81_ECX[11]; }
  static bool TBM(void) { return CPU_Rep._isAMD && CPU_Rep._func_81_ECX[21]; }

  static bool SYSCALL(void) {
    return CPU_Rep._isIntel && CPU_Rep._func_81_EDX[11];
  }
  static bool MMXEXT(void) {
    return CPU_Rep._isAMD && CPU_Rep._func_81_EDX[22];
  }
  static bool RDTSCP(void) {
    return CPU_Rep._isIntel && CPU_Rep._func_81_EDX[27];
  }
  static bool _3DNOWEXT(void) {
    return CPU_Rep._isAMD && CPU_Rep._func_81_EDX[30];
  }
  static bool _3DNOW(void) {
    return CPU_Rep._isAMD && CPU_Rep._func_81_EDX[31];
  }

 private:
  static const InstructionSet_Internal CPU_Rep;

  class InstructionSet_Internal {
   public:
    InstructionSet_Internal()
        : _nIds{0},
          _nExIds{0},
          _isIntel{false},
          _isAMD{false},
          _func_1_ECX{0},
          _func_1_EDX{0},
          _func_7_EBX{0},
          _func_7_ECX{0},
          _func_81_ECX{0},
          _func_81_EDX{0},
          _data{},
          _extdata{} {
      // int cpuInfo[4] = {-1};
      std::array<int, 4> cpui;

      // Calling __cpuid with 0x0 as the function_id argument
      // gets the number of the highest valid function ID.
      __cpuid(cpui.data(), 0);
      _nIds = cpui[0];

      for (int i = 0; i <= _nIds; ++i) {
        __cpuidex(cpui.data(), i, 0);
        _data.push_back(cpui);
      }

      // Capture vendor string
      char vendor[0x20];
      memset(vendor, 0, sizeof(vendor));
      *reinterpret_cast<int*>(vendor) = _data[0][1];
      *reinterpret_cast<int*>(vendor + 4) = _data[0][3];
      *reinterpret_cast<int*>(vendor + 8) = _data[0][2];
      _vendor = vendor;
      if (_vendor == "GenuineIntel") {
        _isIntel = true;
      } else if (_vendor == "AuthenticAMD") {
        _isAMD = true;
      }

      // load bitset with flags for function 0x00000001
      if (_nIds >= 1) {
        _func_1_ECX = _data[1][2];
        _func_1_EDX = _data[1][3];
      }

      // load bitset with flags for function 0x00000007
      if (_nIds >= 7) {
        _func_7_EBX = _data[7][1];
        _func_7_ECX = _data[7][2];
      }

      // Calling __cpuid with 0x80000000 as the function_id argument
      // gets the number of the highest valid extended ID.
      __cpuid(cpui.data(), 0x80000000);
      _nExIds = cpui[0];

      char brand[0x40];
      memset(brand, 0, sizeof(brand));

      for (int i = 0x80000000; i <= _nExIds; ++i) {
        __cpuidex(cpui.data(), i, 0);
        _extdata.push_back(cpui);
      }

      // load bitset with flags for function 0x80000001
      if (_nExIds >= 0x80000001) {
        _func_81_ECX = _extdata[1][2];
        _func_81_EDX = _extdata[1][3];
      }

      // Interpret CPU brand string if reported
      if (_nExIds >= 0x80000004) {
        memcpy(brand, _extdata[2].data(), sizeof(cpui));
        memcpy(brand + 16, _extdata[3].data(), sizeof(cpui));
        memcpy(brand + 32, _extdata[4].data(), sizeof(cpui));
        _brand = brand;
      }
    };

    int _nIds;
    int _nExIds;
    std::string _vendor;
    std::string _brand;
    bool _isIntel;
    bool _isAMD;
    std::bitset<32> _func_1_ECX;
    std::bitset<32> _func_1_EDX;
    std::bitset<32> _func_7_EBX;
    std::bitset<32> _func_7_ECX;
    std::bitset<32> _func_81_ECX;
    std::bitset<32> _func_81_EDX;
    std::vector<std::array<int, 4>> _data;
    std::vector<std::array<int, 4>> _extdata;
  };
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

extern void init_isa_detection() {
}
extern bool __builtin_has_cpufeature_avx2() {
  return InstructionSet::AVX2();
}
extern bool __builtin_has_cpufeature_avx512f() {
  return InstructionSet::AVX512F();
}


}  // namespace gcsort
#endif
