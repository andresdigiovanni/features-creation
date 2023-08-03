import numpy as np

from .base_transformations import Transformations


class CyclicalTransformations(Transformations):
    _ops = ["sin", "cos"]

    def generate_transformations(self, _, columns):
        transformations = {}

        for i_1 in range(len(columns)):
            for op in self._ops:
                col_name = f"{columns[i_1]}__{op}"

                transformations[col_name] = {
                    "column": columns[i_1],
                    "op": op,
                    "type": "cyclical",
                }

        return transformations

    def apply_transformations(self, df, **kwargs):
        column_name, op = kwargs["column"], kwargs["op"]
        column = df[column_name]

        if op == "sin":
            return np.sin(column * (2.0 * np.pi / column).max())

        elif op == "cos":
            return np.cos(column * (2.0 * np.pi / column).max())

        raise Exception("Operation not supported")
