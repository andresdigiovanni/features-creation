from abc import ABC, abstractmethod


class Transformations(ABC):
    @abstractmethod
    def generate_transformations(self, df, columns):
        pass

    @abstractmethod
    def apply_transformations(self, df, **kwargs):
        pass
