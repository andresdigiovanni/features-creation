import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import sem, skew

from .base_transformations import Transformations


class BinsTransformations(Transformations):
    _ops = ["cut", "qcut"]
    _bins = [2, 3, 5, 8, 13]  # default bins to check
    _min_bins = 2
    _max_bins = 20

    def generate_transformations(self, df, columns):
        transformations = {}

        for i_1 in range(len(columns)):
            n_bins_automatic = self._calculate_binning_algorithms_number_of_bins(
                df[columns[i_1]]
            )

            for op in self._ops:
                n_bins_optimizated = [
                    self._calculate_optimizated_number_of_bins(df[columns[i_1]], op)
                ]

                bins = list(set(self._bins + n_bins_automatic + n_bins_optimizated))
                bins = [x for x in bins if self._min_bins <= x <= self._max_bins]

                for n_bins in bins:
                    col_name = f"{columns[i_1]}__{op}_{n_bins}"
                    intervals = self._calculate_intervals(df[columns[i_1]], op, n_bins)

                    transformations[col_name] = {
                        "column": columns[i_1],
                        "op": op,
                        "intervals": intervals,
                        "type": "bins",
                    }

        return transformations

    def _calculate_binning_algorithms_number_of_bins(self, column):
        return [
            self._calculate_doane_number_of_bins(column),
            self._calculate_freedman_diaconis_rule_number_of_bins(column),
            self._calculate_rice_rule_number_of_bins(column),
            self._calculate_scotts_normal_reference_rule_number_of_bins(column),
            self._calculate_square_root_number_of_bins(column),
            self._calculate_sturges_rule_number_of_bins(column),
        ]

    def _calculate_doane_number_of_bins(self, column):
        g1 = skew(column)
        sigma_g1 = sem(column)

        n_bins = 1 + np.log2(len(column)) + np.log2(1 + abs(g1) / sigma_g1)
        return int(n_bins)

    def _calculate_freedman_diaconis_rule_number_of_bins(self, column):
        iqr = np.percentile(column, 75) - np.percentile(column, 25)

        if iqr == 0:
            return 0

        bin_width_fd = 2 * iqr / np.cbrt(len(column))
        n_bins = (column.max() - column.min()) / bin_width_fd

        return int(n_bins)

    def _calculate_rice_rule_number_of_bins(self, column):
        n_bins = round(np.cbrt(len(column)))
        return int(n_bins)

    def _calculate_scotts_normal_reference_rule_number_of_bins(self, column):
        n_bins = np.ceil(3.5 * np.std(column) * len(column) ** (-1 / 3))
        return int(n_bins)

    def _calculate_square_root_number_of_bins(self, column):
        n_bins = np.ceil(np.sqrt(len(column)))
        return int(n_bins)

    def _calculate_sturges_rule_number_of_bins(self, column):
        n_bins = 1 + np.log2(len(column))
        return int(n_bins)

    def _calculate_optimizated_number_of_bins(self, column, op):
        result = minimize_scalar(
            lambda num_bins: self._variance_of_bins(column, op, int(num_bins)),
            bounds=(self._min_bins, self._max_bins),
            method="bounded",
        )

        return int(result.x)

    def _variance_of_bins(self, column, op, num_bins):
        if op == "cut":
            bins = pd.cut(column, bins=num_bins)

        elif op == "qcut":
            bins = pd.qcut(column.rank(method="first"), q=num_bins)

        bin_means = column.groupby(bins).mean()
        return np.var(bin_means)

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

        column_transformed = column_transformed.astype(float)
        return column_transformed
