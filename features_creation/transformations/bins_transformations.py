import pandas as pd

from .base_transformations import Transformations


class BinsTransformations(Transformations):
    _ops = ["cut", "qcut"]
    _bins = [2, 3, 5, 8, 13]

    def generate_transformations(self, df, columns):
        transformations = {}

        for i_1 in range(len(columns)):
            for op in self._ops:
                for bins in self._bins:
                    col_name = f"{columns[i_1]}__{op}_{bins}"
                    intervals = self._calculate_intervals(df[columns[i_1]], op, bins)

                    transformations[col_name] = {
                        "column": columns[i_1],
                        "op": op,
                        "intervals": intervals,
                        "type": "bins",
                    }

        return transformations

    def _calculate_intervals(self, column, op, bins):
        if op == "cut":
            _, intervals = pd.cut(column, bins=bins, retbins=True)

        elif op == "qcut":
            _, intervals = pd.qcut(column.rank(method="first"), q=bins, retbins=True)

        return list(intervals)

    def apply_transformations(self, df, **kwargs):
        column_name, intervals = kwargs["column"], kwargs["intervals"]

        column_transformed = pd.cut(
            df[column_name], bins=intervals, labels=range(len(intervals) - 1)
        )
        return column_transformed
