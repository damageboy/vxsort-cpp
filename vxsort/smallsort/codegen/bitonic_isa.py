from abc import ABC, ABCMeta, abstractmethod

from utils import next_power_of_2


class BitonicISA(ABC, metaclass=ABCMeta):

    @abstractmethod
    def vector_size(self):
        pass

    @abstractmethod
    def max_bitonic_sort_vectors(self):
        pass

    def largest_merge_variant_needed(self):
        return next_power_of_2(self.max_bitonic_sort_vectors());

    @abstractmethod
    def vector_size(self):
        pass

    @abstractmethod
    def vector_type(self):
        pass

    @classmethod
    @abstractmethod
    def supported_types(cls):
        pass

    @abstractmethod
    def generate_prologue(self):
        pass

    @abstractmethod
    def generate_epilogue(self):
        pass


    @abstractmethod
    def generate_1v_basic_sorters(self, ascending: bool):
        pass

    @abstractmethod
    def generate_1v_merge_sorters(self, ascending: bool):
        pass

    def generate_1v_sorters(self, ascending: bool):
        self.generate_1v_basic_sorters(ascending)
        self.generate_1v_merge_sorters(ascending)

    @abstractmethod
    def generate_compounded_sorter(self, width: int, ascending: bool, inline: int):
        pass

    @abstractmethod
    def generate_compounded_merger(self, width: int, ascending: bool, inline: int):
        pass

    @abstractmethod
    def generate_entry_points_full_vectors(self, ascending : bool):
        pass

    @abstractmethod
    def generate_master_entry_point_full(self, ascending : bool):
        pass

    @abstractmethod
    def generate_cross_min_max(self):
        pass

    @abstractmethod
    def generate_strided_min_max(self):
        pass