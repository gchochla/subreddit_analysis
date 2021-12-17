"""Manual nested cross-validation for subreddit classification."""

import argparse
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def parse_args():
    """Parses CL arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--subreddit_repr_path",
        required=True,
        type=str,
        help="path to subreddit posts' embeddings and labels",
    )
    parser.add_argument(
        "-b",
        "--baseline_repr_path",
        required=True,
        type=str,
        help="path to baseline subreddits posts' embeddings and labels",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["all", "both", "embeddings", "loadings"],
        default="all",
        type=str,
        help="what to use as features",
    )
    args = parser.parse_args()
    return args


def load_dataset(
    r_path: str, baseline_path: str, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads pair of embedding files with labels into arrays.

    Args:
        r_path: path to embeddings of subreddit posts.
        baseline_path: path to embeddings of background
            subreddits.
        mode: what features to use basically, "embeddings",
            "loadings" or "both"/"all".

    Returns:
        Tuple of features (N, dim) and labels (N,)
    """

    assert mode in ("embeddings", "loadings", "both", "all")

    df1 = pd.read_csv(r_path, sep=",", index_col="ID")
    df2 = pd.read_csv(baseline_path, sep=",", index_col="ID")
    df = pd.concat([df1, df2])

    print(f"Original shape of data: {df.shape}")
    df.dropna(inplace=True)
    print(f"Shape of data after dropping NaNs: {df.shape}")

    array = df.to_numpy()

    if mode in ("all", "both"):
        X = array[..., :-2]
    elif mode == "embeddings":
        X = array[..., :200]
    else:  # loadings
        X = array[..., 200:-2]
    y = array[..., -2:].astype(int)

    # get rid of other fitness subreddits for now
    X = X[np.logical_or(y[:, 0] == 1, y[:, 1] == 0)]
    y = y[np.logical_or(y[:, 0] == 1, y[:, 1] == 0), 1]

    n0, n1 = len(y[y == 0]), len(y[y == 1])

    print(f"0: {n0}, 1: {n1}, random: {max(n0, n1) / (n1 + n0) * 100: .3f}%")

    return X, y


def nested_cv(
    n_splits: int,
    X: np.ndarray,
    y: np.ndarray,
    n_inner_splits: int,
    model_cls: BaseEstimator,
    parameter_search_space: Dict[str, List[Any]],
):
    """Runs nested cross validation.

    Args:
        n_splits: number of initial splits.
        X: input features.
        y: input labels.
        n_inner_splits: number of splits within the splits.
        model_cls: sklearn classification model.
        parameter_search_space: parameters search space.
    """
    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=n_splits, shuffle=True)

    outer_results = []
    model = model_cls()

    print(f"Running {model.__str__()[:-2]}")  # [:-2] removes parens

    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # standardize data
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=n_inner_splits, shuffle=True)
        # define search
        search = GridSearchCV(
            model,
            parameter_search_space,
            scoring="accuracy",
            cv=cv_inner,
            refit=True,
        )
        # execute search
        result = search.fit(X_train_scaled, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test_scaled)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        print(
            f"\t\t>acc={acc:.3f}, "
            f"est={result.best_score: .3f}, "
            f"cfg={result.best_params_}"
        )
    # summarize the estimated performance of the model
    print(
        f"\tAccuracy: {np.mean(outer_results): .3f} ({np.std(outer_results): .3f})"
    )


if __name__ == "__main__":
    args = parse_args()

    X, y = load_dataset(
        args.subreddit_repr_path, args.baseline_repr_path, args.mode
    )

    nested_cv(10, X, y, 3, RidgeClassifier, {"alpha": [0.1, 0.5, 1, 2, 5]})
    nested_cv(
        10,
        X,
        y,
        3,
        LogisticRegression,
        {"penalty": ["l2", "none"]},
    )
    # takes too long to run, uncomment at your own discretion
    # may require greater "max_iter" to converge
    # nested_cv(10, X, y, 3, LinearSVC, {"C": [0.5, 1, 2], "max_iter": [5000]})
