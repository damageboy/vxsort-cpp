from datetime import datetime

from typing.io import IO
from utils import native_size_map, next_power_of_2
from bitonic_isa import BitonicISA


class AVX512BitonicISA(BitonicISA):
    REMOVE_ME = "<<<REMOVE_ME>>>"

    bitonic_type_map = {
        "i16": "__m512i",
        "u16": "__m512i",
        "i32": "__m512i",
        "u32": "__m512i",
        "f32": "__m512",
        "i64": "__m512i",
        "u64": "__m512i",
        "f64": "__m512d",
    }

    bitonic_mask_map = {
        "i16": "__mmask32",
        "u16": "__mmask32",
        "i32": "__mmask16",
        "u32": "__mmask16",
        "f32": "__mmask16",
        "i64": "__mmask8",
        "u64": "__mmask8",
        "f64": "__mmask8",
    }

    def __init__(self, type: str, f_header: IO):
        self.vector_size_in_bytes = 64

        self.type = type
        self.f_header = f_header

        self.bitonic_size_map = {}

        for t, s in native_size_map.items():
            self.bitonic_size_map[t] = int(self.vector_size_in_bytes / s)

    def clean_print(self, s: str):
        clean_lines = str.join("\n", [l for l in s.splitlines() if not AVX512BitonicISA.REMOVE_ME in l])
        print(clean_lines, file=self.f_header)

    def max_bitonic_sort_vectors(self):
        return 4

    def vector_size(self):
        return self.bitonic_size_map[self.type]

    def vector_type(self):
        return self.bitonic_type_map[self.type]

    def mask_type(self):
        return self.bitonic_mask_map[self.type]


    @classmethod
    def supported_types(cls):
        return __class__.bitonic_type_map.keys()

    def t2d(self, v):
        t = self.type
        if t == "f64":
            return v
        elif t == "f32":
            return f"i2s({v})"
        return f"i2d({v})"

    def i2t(self, v):
        t = self.type
        if t == "f64":
            return f"i2d({v})"
        elif t == "f32":
            return f"i2s({v})"
        return v

    def d2i(self, v):
        t = self.type
        if t == "f64":
            return v
        elif t == "f32":
            raise Exception("WTF")
        return f"d2i({v})"

    def s2i(self, v):
        t = self.type
        if t == "f64":
            raise Exception("WTF")
        elif t == "f32":
            return f"s2i({v})"
        return v

    def generate_param_list(self, start, numParams):
        return str.join(", ", list(map(lambda p: f"d{p:02d}", range(start, start + numParams))))

    def generate_param_def_list(self, numParams):
        t = self.type
        return str.join(", ", list(map(lambda p: f"TV& d{p:02d}", range(1, numParams + 1))))

    def generate_shuffle_X1(self, v):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            return self.i2t(f"_mm512_shuffle_epi8({self.s2i(v)}, x1)")
        if size == 16:
            return self.i2t(f"_mm512_shuffle_epi32({self.s2i(v)}, (_MM_PERM_ENUM)  0b10'11'00'01)")
        elif size == 8:
            return self.d2i(f"_mm512_permute_pd({self.t2d(v)}, (_MM_PERM_ENUM) 0b0'1'0'1'0'1'0'1)")

    def generate_shuffle_X2(self, v):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            return self.i2t(f"_mm512_shuffle_epi32({self.s2i(v)}, (_MM_PERM_ENUM)  0b10'11'00'01)")
        if size == 16:
            return self.i2t(f"_mm512_shuffle_epi32({self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        elif size == 8:
            return self.d2i(f"_mm512_permutex_pd({self.t2d(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")

    def generate_shuffle_X4(self, v):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            return self.i2t(f"_mm512_shuffle_epi32({self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        if size == 16:
            return self.i2t(f"_mm512_permutex_epi64({self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        elif size == 8:
            return self.d2i(f"_mm512_shuffle_f64x2({self.t2d(v)}, {self.t2d(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")

    def generate_shuffle_X8(self, v):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            return self.i2t(f"_mm512_permutex_epi64({self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        if size == 16:
            return self.i2t(f"_mm512_shuffle_i64x2({self.s2i(v)}, {self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        elif size == 8:
            return self.d2i(f"_mm512_shuffle_pd({self.t2d(v)}, {self.t2d(v)}, (_MM_PERM_ENUM) 0xB1)")

    def generate_shuffle_X16(self, v):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            return self.i2t(f"_mm512_shuffle_i64x2({self.s2i(v)}, {self.s2i(v)}, (_MM_PERM_ENUM) 0b01'00'11'10)")
        raise Exception("WTF")

    def generate_min(self, v1, v2):
        t = self.type
        if t == "i16":
            return f"_mm512_min_epi16({v1}, {v2})"
        elif t == "u16":
            return f"_mm512_min_epu16({v1}, {v2})"
        elif t == "i32":
            return f"_mm512_min_epi32({v1}, {v2})"
        elif t == "u32":
            return f"_mm512_min_epu32({v1}, {v2})"
        elif t == "f32":
            return f"_mm512_min_ps({v1}, {v2})"
        elif t == "i64":
            return f"_mm512_min_epi64({v1}, {v2})"
        elif t == "u64":
            return f"_mm512_min_epu64({v1}, {v2})"
        elif t == "f64":
            return f"_mm512_min_pd({v1}, {v2})"

    def generate_max(self, v1, v2):
        t = self.type
        if t == "i16":
            return f"_mm512_max_epi16({v1}, {v2})"
        elif t == "u16":
            return f"_mm512_max_epu16({v1}, {v2})"
        elif t == "i32":
            return f"_mm512_max_epi32({v1}, {v2})"
        elif t == "u32":
            return f"_mm512_max_epu32({v1}, {v2})"
        elif t == "f32":
            return f"_mm512_max_ps({v1}, {v2})"
        elif t == "i64":
            return f"_mm512_max_epi64({v1}, {v2})"
        elif t == "u64":
            return f"_mm512_max_epu64({v1}, {v2})"
        elif t == "f64":
            return f"_mm512_max_pd({v1}, {v2})"

    def generate_mask(self, blend: int, width: int, ascending: bool):
        size = self.vector_size()

        mask = 0
        s = size
        while s > 0:
            mask = mask <<  width | blend
            s -= width

        if not ascending:
            mask = ~mask

        mask = mask & ((1 << size) - 1)
        return mask

    def generate_blended_max(self, src, v1, v2, blend: int, width: int, ascending: bool):
        mask = self.generate_mask(blend, width, ascending)
        t = self.type
        if t == "i16":
            return f"_mm512_mask_max_epi16({src}, 0b{mask:032b}, {v1}, {v2})"
        elif t == "u16":
            return f"_mm512_mask_max_epu16({src}, 0b{mask:032b}, {v1}, {v2})"
        elif t == "i32":
            return f"_mm512_mask_max_epi32({src}, 0b{mask:016b}, {v1}, {v2})"
        elif t == "u32":
            return f"_mm512_mask_max_epu32({src}, 0b{mask:016b}, {v1}, {v2})"
        elif t == "f32":
            return f"_mm512_mask_max_ps({src}, 0b{mask:016b}, {v1}, {v2})"
        elif t == "i64":
            return f"_mm512_mask_max_epi64({src}, 0b{mask:08b}, {v1}, {v2})"
        elif t == "u64":
            return f"_mm512_mask_max_epu64({src}, 0b{mask:08b}, {v1}, {v2})"
        elif t == "f64":
            return f"_mm512_mask_max_pd({src}, 0b{mask:08b}, {v1}, {v2})"

    def get_load_intrinsic(self, v, offset):
        t = self.type
        if t == "f64":
            return f"_mm512_loadu_pd(({t} const *) ((__m512d const *) {v} + {offset}))"
        if t == "f32":
            return f"_mm512_loadu_ps(({t} const *) ((__m512 const *) {v} + {offset}))"
        return f"_mm512_loadu_si512((__m512i const *) {v} + {offset});"

    def get_mask_load_intrinsic(self, v, offset, mask):
        t = self.type

        int_suffix = "bugbugbug"
        max_value = "bugbugbug"
        if self.vector_size() == 8:
            int_suffix = "epi64"
            max_value = f"_mm512_set1_epi64(MAX)"
        elif self.vector_size() == 16:
            int_suffix = "epi32"
            max_value = f"_mm512_set1_epi32(MAX)"
        elif self.vector_size() == 32:
            int_suffix = "epi16"
            max_value = f"_mm512_set1_epi16(MAX)"

        if t == "f64":
            return f"""_mm512_mask_loadu_pd(_mm512_set1_pd(MAX),
                                           {mask},
                                           ({t} const *) ((__m512d const *) {v} + {offset}))"""
        elif t == "f32":
            return f"""_mm512_mask_loadu_ps(_mm512_set1_ps(MAX),
                                           {mask},
                                           ({t} const *) ((__m512 const *) {v} + {offset}))"""

        return f"""_mm512_mask_loadu_{int_suffix}({max_value},
                                              {mask},
                                              ({t} const *) ((__m512i const *) {v} + {offset}))"""

    def get_store_intrinsic(self, ptr, offset, value):
        t = self.type
        if t == "f64":
            return f"_mm512_storeu_pd(({t} *) ((__m512d *)  {ptr} + {offset}), {value})"
        if t == "f32":
            return f"_mm512_storeu_ps(({t} *) ((__m512 *)  {ptr} + {offset}), {value})"
        return f"_mm512_storeu_si512((__m512i *) {ptr} + {offset}, {value})"

    def get_mask_store_intrinsic(self, ptr, offset, value, mask):
        t = self.type

        int_suffix = "bugbugbug"
        if self.vector_size() == 8:
            int_suffix = "epi64"
        elif self.vector_size() == 16:
            int_suffix = "epi32"
        elif self.vector_size() == 32:
            int_suffix = "epi32"

        if t == "f64":
            return f"_mm512_mask_storeu_pd(({t} *) ((__m512d *)  {ptr} + {offset}), {mask}, {value})"
        if t == "f32":
            return f"_mm512_mask_storeu_ps(({t} *) ((__m512 *)  {ptr} + {offset}), {mask}, {value})"
        return f"_mm512_mask_storeu_{int_suffix}((__m512i *) {ptr} + {offset}, {mask}, {value})"

    def generate_x1_epi16_shuffle_vec(self):
        if self.type == "u16" or self.type == "i16":
            l = [None]*8
            l[0] = 0x0504070601000302
            for i in range(1, 8):
                l[i] = l[i-1] + 0x0808080808080808
            return f"const TV x1 = _mm512_set_epi64(0x{l[3]:08X}, 0x{l[2]:08X}, 0x{l[1]:08X}, 0x{l[0]:08X}," "\n" \
                   f"                               0x{l[7]:08X}, 0x{l[6]:08X}, 0x{l[5]:08X}, 0x{l[4]:08X})"

        return AVX512BitonicISA.REMOVE_ME

    def autogenerated_blabber(self):
        return f"""/////////////////////////////////////////////////////////////////////////////
////
// This file was auto-generated by a tool at {datetime.now().strftime("%F %H:%M:%S")}
//
// It is recommended you DO NOT directly edit this file but instead edit
// the code-generator that generated this source file instead.
/////////////////////////////////////////////////////////////////////////////"""

    def generate_prologue(self):
        t = self.type
        self.clean_print(f"""{self.autogenerated_blabber()}

#ifndef BITONIC_MACHINE_AVX512_{t.upper()}_H
#define BITONIC_MACHINE_AVX512_{t.upper()}_H

#include "../../vxsort_targets_enable_avx512.h"

#include <limits>
#include <immintrin.h>
#include "../bitonic_machine.h"

#define i2d _mm512_castsi512_pd
#define d2i _mm512_castpd_si512
#define i2s _mm512_castsi512_ps
#define s2i _mm512_castps_si512
#define s2d _mm512_castps_pd
#define d2s _mm521_castpd_ps

namespace vxsort {{
namespace smallsort {{
using namespace vxsort::types;

template<> struct bitonic_machine<{t}, AVX512> {{
    static const i32 N = {self.vector_size()};
    static constexpr {t} MAX = std::numeric_limits<{t}>::max();
public:
    typedef {self.vector_type()} TV;
    typedef {self.mask_type()} TMASK;

""")

    def generate_epilogue(self):
        self.clean_print(f"""
}};
}}
}}

#undef i2d
#undef d2i
#undef i2s
#undef s2i
#undef s2d
#undef d2s

#include "../../vxsort_targets_disable.h"

#endif
""")

    def generate_1v_basic_sorters(self, asc: bool):
        g = self
        suffix = "ascending" if asc else "descending"

        self.clean_print(f"""    static INLINE void sort_01v_{suffix}({g.generate_param_def_list(1)}) {{
        TV  min, s;
        {g.generate_x1_epi16_shuffle_vec()};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b0110, 4, asc)};

        s = {g.generate_shuffle_X2("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b00111100, 8, asc)};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b01011010, 8, asc)};

        s = {g.generate_shuffle_X4("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b0000111111110000, 16, asc)};

        s = {g.generate_shuffle_X2("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b0011001111001100, 16, asc)};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b0101010110101010, 16, asc)};""")

        if g.vector_size() >= 16:
            self.clean_print(f"""
        s = {g.generate_shuffle_X8("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b00000000111111111111111100000000, 32, asc)};

        s = {g.generate_shuffle_X4("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b00001111000011111111000011110000, 32, asc)};

        s = {g.generate_shuffle_X2("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b00110011001100111100110011001100, 32, asc)};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b01010101010101011010101010101010, 32, asc)};""")

        if g.vector_size() >= 32:
            self.clean_print(f"""
        s = {g.generate_shuffle_X16("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b11111111111111110000000000000000, 32, asc)};

        s = {g.generate_shuffle_X8("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b1111111100000000, 16, asc)};

        s = {g.generate_shuffle_X4("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b11110000, 8, asc)};

        s = {g.generate_shuffle_X2("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b1100, 4, asc)};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b10, 2, asc)};""")

        self.clean_print("    }")

    def generate_1v_merge_sorters(self, asc: bool):
        g = self
        type = self.type
        suffix = "ascending" if asc else "descending"

        g.clean_print(f"""    static INLINE void merge_01v_{suffix}({g.generate_param_def_list(1)}) {{
        TV  min, s;
        {g.generate_x1_epi16_shuffle_vec()};""")

        if g.vector_size() >= 32:
            g.clean_print(f"""
        s = {g.generate_shuffle_X16("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b11111111111111110000000000000000, 16, asc)};""")

        if g.vector_size() >= 16:
            g.clean_print(f"""
        s = {g.generate_shuffle_X8("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b1111111100000000, 16, asc)};""")

        g.clean_print(f"""
        s = {g.generate_shuffle_X4("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b11110000, 8, asc)};

        s = {g.generate_shuffle_X2("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b1100, 4, asc)};

        s = {g.generate_shuffle_X1("d01")};
        min = {g.generate_min("s", "d01")};
        d01 = {g.generate_blended_max("min", "s", "d01", 0b10, 2, asc)};""")

        g.clean_print("    }")

    def generate_compounded_sorter(self, width: int, asc: bool, inline: int):
        g = self
        w1 = int(next_power_of_2(width) / 2)
        w2 = int(width - w1)

        suffix = "ascending" if asc else "descending"
        rev_suffix = "descending" if asc else "ascending"

        inl = "INLINE" if inline else "NOINLINE"

        g.clean_print(f"""    static {inl} void sort_{width:02d}v_{suffix}({g.generate_param_def_list(width)}) {{
        TV tmp;

        sort_{w1:02d}v_{suffix}({g.generate_param_list(1, w1)});
        sort_{w2:02d}v_{rev_suffix}({g.generate_param_list(w1 + 1, w2)});""")

        for r in range(w1 + 1, width + 1):
            x = w1 + 1 - (r - w1)
            g.clean_print(f"""
        tmp = d{r:02d};
        d{r:02d} = {g.generate_max(f"d{x:02d}", f"d{r:02d}")};
        d{x:02d} = {g.generate_min(f"d{x:02d}", "tmp")};""")

        g.clean_print(f"""
        merge_{w1:02d}v_{suffix}({g.generate_param_list(1, w1)});
        merge_{w2:02d}v_{suffix}({g.generate_param_list(w1 + 1, w2)});""")
        g.clean_print("    }")

    def generate_compounded_merger(self, width: int, asc: bool, inline: int):
        g = self

        w1 = int(next_power_of_2(width) / 2)
        w2 = int(width - w1)

        suffix = "ascending" if asc else "descending"

        inl = "INLINE" if inline else "NOINLINE"

        g.clean_print(f"""    static {inl} void merge_{width:02d}v_{suffix}({g.generate_param_def_list(width)}) {{
        TV tmp;""")

        for r in range(w1 + 1, width + 1):
            x = r - w1
            g.clean_print(f"""
        tmp = d{x:02d};
        d{x:02d} = {g.generate_min(f"d{r:02d}", f"d{x:02d}")};
        d{r:02d} = {g.generate_max(f"d{r:02d}", "tmp")};""")

        g.clean_print(f"""
        merge_{w1:02d}v_{suffix}({g.generate_param_list(1, w1)});
        merge_{w2:02d}v_{suffix}({g.generate_param_list(w1 + 1, w2)});""")
        g.clean_print("    }")

    def generate_reverse(self, v: str):
        t = self.type
        size = self.bitonic_size_map[t]
        if size == 32:
            s1 = f"_mm512_shuffle_epi8({self.s2i(v)}, x1)"
            s1 = f"_mm512_shuffle_epi32({s1}, (_MM_PERM_ENUM) 0b00'01'10'11)"
            return self.i2t(f"_mm512_shuffle_i64x2({s1}, {s1}, (_MM_PERM_ENUM) 0b00'01'10'11)")
        elif size == 16:
            s1 = f"_mm512_shuffle_epi32({self.s2i(v)}, (_MM_PERM_ENUM) 0b00'01'10'11)"
            return self.i2t(f"_mm512_shuffle_i64x2({s1}, {s1}, (_MM_PERM_ENUM) 0b00'01'10'11)")
        elif size == 8:
            s1 = f"d2i(_mm512_permute_pd({self.t2d(v)}, (_MM_PERM_ENUM) 0b0'1'0'1'0'1'0'1))"
            return self.i2t(f"_mm512_shuffle_i64x2({s1}, {s1}, (_MM_PERM_ENUM) 0b00'01'10'11)")

    def generate_cross_min_max(self):
        g = self

        g.clean_print(f"""    static INLINE void cross_min_max(TV& d01, TV& d02) {{
        TV tmp;
        {g.generate_x1_epi16_shuffle_vec()};

        tmp = {g.generate_reverse("d02")};
        d02 = {g.generate_max("d01", "tmp")};
        d01 = {g.generate_min("d01", "tmp")};""")
        g.clean_print("    }")

    def generate_strided_min_max(self):
        g = self

        g.clean_print(f"""    static INLINE void strided_min_max(TV& d01, TV& d02) {{
        TV tmp;

        tmp = d01;
        d01 = {g.generate_min("d02", "d01")};
        d02 = {g.generate_max("d02", "tmp")};""")
        g.clean_print("    }")

    def generate_entry_points_full_vectors(self, asc: bool):
        type = self.type
        g = self
        sfx = "ascending" if asc else "descending"
        for m in range(1, g.max_bitonic_sort_vectors() + 1):
            g.clean_print(f"""
    // This is generated for testing purposes only
    static NOINLINE void sort_{m:02d}v_full_{sfx}({type} *ptr) {{""")

            for l in range(0, m):
                g.clean_print(f"        TV d{l + 1:02d} = {g.get_load_intrinsic('ptr', l)};")

            g.clean_print(f"        sort_{m:02d}v_{sfx}({g.generate_param_list(1, m)});")

            for l in range(0, m):
                g.clean_print(f"        {g.get_store_intrinsic('ptr', l, f'd{l + 1:02d}')};")

            g.clean_print("    }")

    def generate_entry_points_partial_vectors(self):
        type = self.type
        g = self
        for m in range(1, g.max_bitonic_sort_vectors() + 1):
            g.clean_print(f"""
    static NOINLINE void sort_{m:02d}v_partial({type} *ptr, i32 remainder) {{
        const auto mask = 0x{((1 << self.vector_size()) - 1):X} >> ((N - remainder) & (N-1));
""")

            for l in range(0, m-1):
                g.clean_print(f"        TV d{l + 1:02d} = {g.get_load_intrinsic('ptr', l)};")

            g.clean_print(f"        TV d{m:02d} = {g.get_mask_load_intrinsic('ptr', m - 1, 'mask')};")

            g.clean_print(f"        sort_{m:02d}v_ascending({g.generate_param_list(1, m)});")

            for l in range(0, m-1):
                g.clean_print(f"        {g.get_store_intrinsic('ptr', l, f'd{l + 1:02d}')};")

            g.clean_print(f"        {g.get_mask_store_intrinsic('ptr', m - 1, f'd{m:02d}', 'mask')};")

            g.clean_print("    }")


    def generate_master_entry_point_full(self, asc : bool):
        t = self.type
        g = self
        sfx = "ascending" if asc else "descending"

        g.clean_print(f"""
    // This is generated for testing purposes only
    static NOINLINE void sort_full_vectors_{sfx}({t} *ptr, usize length) {{
        assert(length % N == 0);
        switch(length / N) {{""")

        for m in range(1, self.max_bitonic_sort_vectors() + 1):
            g.clean_print(f"            case {m}: sort_{m:02d}v_full_{sfx}(ptr); break;")
        g.clean_print("        }")
        g.clean_print("    }")


    #     s = f"""void vxsort::smallsort::bitonic<{t}, vector_machine::AVX512 >::sort({t} *ptr, size_t length) {{
    # const auto fullvlength = length / N;
    # const i32 remainder = (int) (length - fullvlength * N);
    # const auto v = fullvlength + ((remainder > 0) ? 1 : 0);
    # switch(v) {{"""
    #     print(s, file=f_src)
    #
    #     for m in range(1, self.max_bitonic_sort_vectors() + 1):
    #         s = f"        case {m}: sort_{m:02d}v_alt(ptr, remainder); break;"
    #         print(s, file=f_src)
    #     print("    }", file=f_src)
    #
    #     print("}", file=f_src)
    #     pass
