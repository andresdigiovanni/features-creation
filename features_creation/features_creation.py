from itertools import islice

import numpy as np
import pandas as pd
from tqdm import tqdm

from .transformations import (
    BinsTransformations,
    CyclicalTransformations,
    MathTransformations,
)


class FeaturesCreation:
    """A class for creating new features using various transformations."""

    def __init__(self) -> None:
        self._transformations = {
            "bins": BinsTransformations(),
            "cyclical": CyclicalTransformations(),
            "math": MathTransformations(),
        }

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        clf,
        n_new_features: int,
        verbose: bool = True,
    ) -> dict:
        """Fit the feature creation model and generate new features.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            y (pd.Series): The target variable.
            clf: The classifier used for feature importance evaluation.
            n_new_features (int): The number of new features to generate.
            verbose (bool, optional): If True, display progress information. Defaults to True.

        Returns:
            dict: A dictionary containing the generated transformations for new features.
        """
        numerical_columns = self._get_numerical_columns(df)
        transformations = self._generate_transformations(df, numerical_columns)

        if verbose:
            print(
                "Created {} candidate features in {} columns".format(
                    len(transformations), len(numerical_columns)
                )
            )

        while len(transformations) > n_new_features:
            list_of_chunks = self._create_chunks(transformations, n_new_features * 10)

            for chunk in tqdm(list_of_chunks, disable=not verbose):
                feature_importances = self._most_importances_features(df, y, clf, chunk)
                to_remove = feature_importances[n_new_features:]
                transformations = {
                    k: v for k, v in transformations.items() if k not in to_remove
                }

        return transformations

    def _get_numerical_columns(self, df):
        return list(df.select_dtypes(include=[np.number]).columns)

    def _generate_transformations(self, df, columns):
        transformations = {
            key: transformer.generate_transformations(df, columns)
            for key, transformer in self._transformations.items()
        }

        return dict(sum((list(t.items()) for t in transformations.values()), []))

    def _create_chunks(self, items, size):
        result = []
        it = iter(items)
        for _ in range(0, len(items), size):
            result.append({k: items[k] for k in islice(it, size)})

        return result

    def _most_importances_features(self, df, y, clf, transformations):
        x_train = self.apply_transformation(df, transformations)

        clf.fit(x_train, y)
        feature_importances = clf.feature_importances_
        importances_with_names = list(zip(x_train.columns, feature_importances))
        importances_with_names.sort(key=lambda x: x[1], reverse=True)
        feature_importances = [name for name, _ in importances_with_names]

        return feature_importances

    def apply_transformation(
        self, df: pd.DataFrame, transformations: dict
    ) -> pd.DataFrame:
        """Apply the transformations to the DataFrame and create new features.

        Args:
            df (pd.DataFrame): The input DataFrame containing the data.
            transformations (dict): A dictionary of transformations for new features.

        Returns:
            pd.DataFrame: The DataFrame with new features created by applying the transformations.
        """
        list_transformations = []
        list_columns = []

        for k, v in transformations.items():
            type_ = v["type"]

            list_transformations.append(
                self._transformations[type_].apply_transformations(df, **v)
            )
            list_columns.append(k)

        new_df = pd.concat(list_transformations, axis=1)
        new_df.columns = list_columns
        new_df.index = df.index

        return new_df
