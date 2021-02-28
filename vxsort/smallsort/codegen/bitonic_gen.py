#!/usr/bin/env python3
import argparse
import os
from enum import Enum

from typing.io import IO

from avx2 import AVX2BitonicISA
from avx512 import AVX512BitonicISA
from neon import NeonBitonicISA
from bitonic_isa import BitonicISA

BitonicISA.register(AVX2BitonicISA)
BitonicISA.register(AVX512BitonicISA)
BitonicISA.register(NeonBitonicISA)



def get_generator_supported_types(vector_isa):
    if isinstance(vector_isa, str):
        vector_isa = VectorISA[vector_isa]
    if vector_isa == VectorISA.AVX2:
        return AVX2BitonicISA.supported_types()
    elif vector_isa == VectorISA.AVX512:
        return AVX512BitonicISA.supported_types()
    elif vector_isa == VectorISA.NEON:
        return NeonBitonicISA.supported_types()
    else:
        raise Exception(f"Non-supported vector machine-type: {vector_isa}")


def get_generator(vector_isa, type, f_header: IO):
    if isinstance(vector_isa, str):
        vector_isa = VectorISA[vector_isa]
    if vector_isa == VectorISA.AVX2:
        return AVX2BitonicISA(type, f_header)
    elif vector_isa == VectorISA.AVX512:
        return AVX512BitonicISA(type, f_header)
    elif vector_isa == VectorISA.NEON:
        return NeonBitonicISA(type, f_header)
    else:
        raise Exception(f"Non-supported vector machine-type: {vector_isa}")


def generate_per_type(f_header: IO, type, vector_isa, break_inline):
    g = get_generator(vector_isa, type, f_header)
    g.generate_prologue()
    g.generate_1v_sorters(ascending=True)
    g.generate_1v_sorters(ascending=False)
    for width in range(2, g.max_bitonic_sort_vectors() + 1):
        # Allow breaking the inline chain once in a while (configurable)
        if break_inline == 0 or width % break_inline != 0:
            inline = True
        else:
            inline = False
        g.generate_compounded_sorter(width, asc=True, inline=inline)
        g.generate_compounded_sorter(width, asc=False, inline=inline)
        if width <= g.largest_merge_variant_needed():
            g.generate_compounded_merger(width, asc=True, inline=inline)
            g.generate_compounded_merger(width, asc=False, inline=inline)


    g.generate_cross_min_max()
    g.generate_strided_min_max()

    print("\n#ifdef BITONIC_TESTS", file=f_header)
    g.generate_entry_points_full_vectors(asc=True)
    g.generate_entry_points_full_vectors(asc=False)
    g.generate_master_entry_point_full(asc=True)
    g.generate_master_entry_point_full(asc=False)
    print("\n#endif", file=f_header)
    g.generate_epilogue()


class Language(Enum):
    csharp = 'csharp'
    cpp = 'cpp'
    rust = 'rust'

    def __str__(self):
        return self.value


class VectorISA(Enum):
    AVX2 = 'AVX2'
    AVX512 = 'AVX512'
    NEON = 'NEON'
    SVE = 'SVE'

    def __str__(self):
        return self.value

def generate_all_types():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--language", type=Language, choices=list(Language),
    #                    help="select output language: csharp/cpp/rust")
    parser.add_argument("--vector-isa",
                        nargs='+',
                        default='all',
                        help='list of vector ISA to generate',
                        choices=list(VectorISA).append("all"))
    parser.add_argument("--break-inline", type=int, default=0, help="break inlining every N levels")

    parser.add_argument("--output-dir", type=str,
                        help="output directory")

    opts = parser.parse_args()

    if 'all' in opts.vector_isa:
        opts.vector_isa = list(VectorISA)

    for isa in opts.vector_isa:
        for t in get_generator_supported_types(isa):
            filename = f"bitonic_machine.{isa}.{t}.generated"
            print(f"Generating {filename}.{{h,.cpp}}")
            dirname = os.path.join(opts.output_dir, isa.lower())
            os.makedirs(dirname, exist_ok=True)
            h_filename = os.path.join(dirname, filename + ".h")
            with open(h_filename, "w") as f_header:
                generate_per_type(f_header, t, isa, opts.break_inline)

if __name__ == '__main__':
    generate_all_types()
