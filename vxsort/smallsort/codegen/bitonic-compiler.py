#!/usr/bin/env python3
from __future__ import annotations
import copy
from dataclasses import dataclass
from enum import Enum
from typing import override


from functional import seq
from tabulate import tabulate


class top_bottom_ind(Enum):
    Top = (0,)
    Bottom = (1,)


class vector_machine(Enum):
    AVX2 = (1,)
    AVX512 = (2,)


class primitive_type(Enum):
    i16 = (2,)
    i32 = (4,)
    i64 = (8,)
    f32 = (4,)
    f64 = 8


width_dict = {
    vector_machine.AVX2: 32,
    vector_machine.AVX512: 64,
}


class BitonicStage:
    def __init__(self, stage: int, pairs: list[tuple[int, int]]):
        self.stage = stage
        self.pairs = pairs

    @override
    def __repr__(self):
        return f"S{self.stage}: {self.pairs}"


class BitonicSorter:
    stages: dict[int, list[tuple[int, int]]]

    def __init__(self, n: int):
        self.stages = {}
        _ = self.generate_bitonic_sorter(n)

    # Bitonic sorters are recursive in nature, where we sort both halves of the input
    # and proceed to merge to two halves via a bitonic merge operation.
    def generate_bitonic_sorter(self, n: int, stage: int = 0, i: int = 0) -> int:
        if n == 1:
            return stage

        k = n // 2
        _ = self.generate_bitonic_sorter(k, stage, i)
        stage = self.generate_bitonic_sorter(k, stage, i + k)
        return self.generate_bitonic_merge(n, stage, i, True)

    def generate_bitonic_merge(self, n: int, stage: int, i: int, initial_merge: bool) -> int:
        if n == 1:
            return stage
        k = n // 2

        if initial_merge:
            stage_pairs = seq.range(i, i + k).zip(seq.range(i + k, i + n).reverse()).to_list()
        else:
            stage_pairs = seq.range(i, i + k).map(lambda x: (x, x + k)).to_list()

        self.add_ops(BitonicStage(stage, stage_pairs))

        _ = self.generate_bitonic_merge(k, stage + 1, i, False)
        return self.generate_bitonic_merge(k, stage + 1, i + k, False)

    def add_ops(self, bs: BitonicStage):
        if not bs.stage in self.stages:
            self.stages[bs.stage] = bs.pairs
        else:
            self.stages[bs.stage].extend(bs.pairs)


class ShuffleOps:
    def __init__(self):
        pass


@dataclass
class StageVector:
    vecid: int
    data: list[int]


@dataclass
class StageVectors:
    top: list[StageVector]
    bot: list[StageVector]


@dataclass
class VecDist:
    v: int
    e: int


def is_single_vector_shuffle(input, next_stage):
    pass


class VectorizedStage:
    input: StageVectors
    output: StageVectors

    def __init__(
        self,
        elem_width: int,
        prev: VectorizedStage | None = None,
        stage: list[tuple[int, int]] | None = None,
        shuffels: list[ShuffleOps] | None = None,
    ):
        self.shuffles = shuffels
        self.apply_minmax()


class BitonicVectorizer:
    def __init__(
        self,
        stages: dict[int, list[tuple[int, int]]],
        type: primitive_type,
        vm: vector_machine,
    ):
        self.stages = stages
        self.type = type
        self.vm = vm
        self.elem_width = int(width_dict[vm] / int(type.value[0]))
        self.vectorized_stages = {}
        self.process_stages()

    def process_stages(self):
        flat_stages = seq.range(len(self.stages)).map(lambda x: self.stages[x]).to_list()

        self.vectorized_stages = []

        prev = None
        for cur in flat_stages:
            vec_stage = VectorizedStage(self.elem_width, prev, cur)
            self.vectorized_stages.append(vec_stage)
            prev = vec_stage


def generate_bitonic_sorter(num_vecs: int, type: primitive_type, vm: vector_machine):
    total_elements = int(num_vecs * (width_dict[vm] / int(type.value[0])))

    print(f"Building {vm} sorter for {total_elements} elements")

    # Generate the list of pairs to be compared per stage
    # each stage is a list of pairs tha can be compared in parallel

    bitonic_sorter = BitonicSorter(total_elements)

    bitonic_vectorizer = BitonicVectorizer(bitonic_sorter.stages, type, vm)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    generate_bitonic_sorter(4, primitive_type.i32, vector_machine.AVX2)
