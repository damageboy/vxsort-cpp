from __future__ import annotations

from typing import override

from functional import seq


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

    def generate_bitonic_merge(
        self, n: int, stage: int, i: int, initial_merge: bool
    ) -> int:
        if n == 1:
            return stage
        k = n // 2

        if initial_merge:
            stage_pairs = (
                seq.range(i + 1, i + k + 1)
                .zip(seq.range(i + k + 1, i + n + 1).reverse())
                .to_list()
            )
        else:
            stage_pairs = (
                seq.range(i + 1, i + k + 1).map(lambda x: (x, x + k)).to_list()
            )

        self.add_ops(BitonicStage(stage, stage_pairs))

        _ = self.generate_bitonic_merge(k, stage + 1, i, False)
        return self.generate_bitonic_merge(k, stage + 1, i + k, False)

    def add_ops(self, bs: BitonicStage):
        if bs.stage not in self.stages:
            self.stages[bs.stage] = bs.pairs
        else:
            self.stages[bs.stage].extend(bs.pairs)
