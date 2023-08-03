import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

from features_creation import FeaturesCreation

# get data
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# set up
fe_cr = FeaturesCreation()
clf = LGBMClassifier(verbose=-1)
n_new_features = 4
x, y = (
    iris_df.drop(columns=["target"]),
    iris_df["target"],
)

# create new transformations
transformations = fe_cr.fit(
    x,
    y,
    clf,
    n_new_features,
    verbose=True,
)

# apply transformations
transformed_df = fe_cr.apply_transformation(iris_df, transformations)
iris_transformed_df = pd.concat([iris_df, transformed_df], axis=1)

print(iris_transformed_df.head())
