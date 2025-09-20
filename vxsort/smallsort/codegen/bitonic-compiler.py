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
        self.elem_width = elem_width
        if not prev:
            self.input = StageVectors(*self.break_into_vectors(stage))
            self.output = copy.deepcopy(self.input)
            self.print_output()
        else:
            self.input = prev.output
            next_stage = StageVectors(*self.break_into_vectors(stage))
            self.output = self.generate_shuffles(self.input, next_stage)
            self.print_output()

        self.shuffles = shuffels
        self.apply_minmax()

    def apply_minmax(self):
        for i, (top_vec, bot_vec) in enumerate(zip(self.output.top, self.output.bot)):
            for j, (t, b) in enumerate(zip(top_vec.data, bot_vec.data)):
                if t > b:
                    self.output.top[i].data[j] = b
                    self.output.bot[i].data[j] = t

    def break_into_vectors(self, cur: list[tuple[int, int]]):
        top = seq(cur).map(lambda x: x[0]).to_list()
        bot = seq(cur).map(lambda x: x[1]).to_list()
        top_vectors = seq(self.chunk_to_vectors(top)).enumerate().map(lambda x: StageVector(x[0], x[1])).to_list()
        o = len(top_vectors)
        bot_vectors = seq(self.chunk_to_vectors(bot)).enumerate().map(lambda x: StageVector(x[0] + o, x[1])).to_list()

        return top_vectors, bot_vectors

    def chunk_to_vectors(self, data):
        return [data[x : x + self.elem_width] for x in range(0, len(data), self.elem_width)]

    def tb_str(self, tb: int):
        if tb == 0:
            return "top"
        else:
            return "bot"

    def generate_shuffles(self, input, next_stage):
        # We support a few prototypes of shuffles that should suffice for
        # all mutating the input vectors into the output shape before applying a
        # min/max operation. which is, in itself, can be thought of as a cross vector shuffle/blend
        # operation.
        # The prototypes are:
        # * One-vector shuffle: At least one element of each pair in the next-stage is
        #                       already on *one* of the input vectors, but never both
        #                       on the same input vector.
        #                       In this case, it is enough to perform a single vector shuffle
        #                       to place all the pairs "in-front" of each other and perform a
        #                       min/max operation on the vector.

        if is_single_vector_shuffle(input, next_stage):
            return perform_single_vector_shuffle(input, next_stage)

        # top_str = ""
        # bot_str = ""
        # for i, (top_vec, bot_vec) in enumerate(zip(shuffled_vectors.top, shuffled_vectors.bot)):
        #     for j, (t, b) in enumerate(zip(top_vec.data, bot_vec.data)):
        #         tb, v_idx, v_pos, = self.find_index(input, (t, b))
        #         top_dist = VecDist(i - v_idx[0], j - v_pos[0])
        #         bot_dist = VecDist(i - v_idx[1], j - v_pos[1])
        #         top_str += f"T: {t} ({self.tb_str(tb[0])}, {v_idx[0]}/{v_pos[0]}) <-> (top, {i}/{j}) => {top_dist}\n"
        #         bot_str += f"B: {b} ({self.tb_str(tb[1])}, {v_idx[1]}/{v_pos[1]}) <-> (bot, {i}/{j}) => {bot_dist}\n"
        # print(top_str)
        # print(bot_str)

    def print_output(self):
        table = tabulate(
            [
                seq(self.output.top)
                .map(
                    lambda v: [
                        v.vecid,
                        tabulate([v.data], tablefmt="rounded_outline", intfmt="2d"),
                    ]
                )
                .flatten()
                .to_list(),
                seq(self.output.bot)
                .map(
                    lambda v: [
                        v.vecid,
                        tabulate([v.data], tablefmt="rounded_outline", intfmt="2d"),
                    ]
                )
                .flatten()
                .to_list(),
            ],
            tablefmt="fancy_grid",
        )

        print(table)

    def find_index(self, input, indices: tuple[int, int]):
        top_bottom: list[int] = [0, 0]
        vec_idx: list[int] = [0, 0]
        vec_pos: list[int] = [0, 0]

        for k, x in enumerate(indices):
            for top_vec, bot_vec in zip(input.top, input.bot):
                found = False
                for j, (t, b) in enumerate(zip(top_vec.data, bot_vec.data)):
                    if x in (t, b):
                        top_bottom[k] = 0 if x == t else 1
                        vec_idx[k] = top_vec.vecid if x == t else bot_vec.vecid
                        vec_pos[k] = j
                        found = True
                        break
                if found:
                    break

        return top_bottom, vec_idx, vec_pos


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
