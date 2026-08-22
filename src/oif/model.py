"""The object-based attention (OBA) model.

A linear model of how many fixations an object attracts, given where it sits
in the scene and how big, how far and how salient it is::

    log1p(fixations) ~ log1p(size) + log1p(depth) + z(eccentricity) + z(salience)

Fitted on the OiF objects this accounts for most of the variance in
cumulative fixations per object (R^2 = 0.82 in the paper); fitted on
COCO-Freeview it does less well (R^2 = 0.66), and a single model pooled
across both does worse still unless a dataset term is included. Those
numbers are the reference points to beat, not a ceiling - the point of
shipping the model is to make it easy to beat.

Implemented with plain least squares (numpy), so the package has no hard
dependency on statsmodels or scikit-learn. If statsmodels is installed,
``summary()`` gives the usual inferential table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["ObjectAttentionModel", "DEFAULT_TERMS", "PUBLISHED_FIT"]

DEFAULT_TERMS: Sequence[str] = ("log_size", "log_depth", "z_ecc", "z_salience")

#: Published model fit statistics, for reference when comparing your own
#: numbers. Source: Hall & Loh, "Objects in Focus: Predicting Object-Based
#: Attention from Spatial Features", Table 1.
PUBLISHED_FIT: Dict[str, Dict[str, float]] = {
    "oif":          {"n_objects": 2493,  "r2": 0.82, "f": 2920.5,
                     "log_mae_object": 0.50, "mae_object": 35.4, "mae_scene": 46.9},
    "coco":         {"n_objects": 31138, "r2": 0.66, "f": 1530.7,
                     "log_mae_object": 0.59, "mae_object": 4.8, "mae_scene": 7.1},
    "combined":     {"n_objects": 33631, "r2": 0.49, "f": 7945.6,
                     "log_mae_object": 0.76, "mae_object": 11.5, "mae_scene": 12.0},
    "combined_ds":  {"n_objects": 33631, "r2": 0.73, "f": 18330.0,
                     "log_mae_object": 0.59, "mae_object": 8.6, "mae_scene": 10.1},
}


@dataclass
class ObjectAttentionModel:
    """Least-squares model of fixations per object.

    Examples
    --------
    >>> from oif import OiF, ObjectAttentionModel
    >>> table = ...  # object features joined to fixation counts
    >>> model = ObjectAttentionModel().fit(table)
    >>> table["predicted"] = model.predict(table)
    >>> model.score(table)["r2"]
    """

    terms: Sequence[str] = field(default_factory=lambda: list(DEFAULT_TERMS))
    outcome: str = "log_sum"
    fit_intercept: bool = True
    coef_: Optional[np.ndarray] = None
    intercept_: float = 0.0
    used_terms_: List[str] = field(default_factory=list)
    n_obs_: int = 0

    # -- fitting -----------------------------------------------------------
    def _design(self, df: pd.DataFrame, terms: Sequence[str]) -> np.ndarray:
        X = df[list(terms)].to_numpy(dtype=float)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None
            ) -> ObjectAttentionModel:
        """Fit on a table that already carries the transformed columns.

        Run :func:`oif.features.add_model_terms` first. Terms absent from the
        table are dropped with the fit recorded in ``used_terms_``, so a
        dataset with no salience maps still fits the remaining three.
        """
        present = [t for t in self.terms if t in df.columns]
        # A term that is entirely missing (no salience maps, say) would empty
        # the table on dropna. Leave it out and say so, rather than fitting
        # on nothing.
        terms = [t for t in present if df[t].notna().any()]
        dropped = [t for t in present if t not in terms]
        if dropped:
            import warnings
            warnings.warn(
                f"dropping model term(s) {dropped}: no values in the table. "
                "Fitting on " + (", ".join(terms) if terms else "nothing"),
                stacklevel=2,
            )
        if not terms:
            raise ValueError(
                f"none of the model terms {list(self.terms)} are usable; columns "
                f"present: {present or 'none'}. Call "
                "oif.features.add_model_terms(table) first, and check that the "
                "table has fixation counts."
            )
        if self.outcome not in df.columns:
            raise ValueError(f"outcome column {self.outcome!r} not in the table")

        cols = terms + [self.outcome]
        data = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
        X = self._design(data, terms)
        y = data[self.outcome].to_numpy(dtype=float)
        if sample_weight is not None:
            w = np.sqrt(np.asarray(sample_weight, dtype=float)[data.index])
            X, y = X * w[:, None], y * w

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        else:
            self.intercept_, self.coef_ = 0.0, beta
        self.used_terms_ = terms
        self.n_obs_ = int(len(data))
        return self

    # -- use ---------------------------------------------------------------
    def predict(self, df: pd.DataFrame, scale: str = "log") -> np.ndarray:
        """Predict per-object fixations.

        ``scale="log"`` returns the modelled ``log1p`` value; ``"count"``
        back-transforms with ``expm1`` to fixation counts.
        """
        if self.coef_ is None:
            raise RuntimeError("model is not fitted - call fit() first")
        X = df[self.used_terms_].to_numpy(dtype=float)
        pred = X @ self.coef_ + self.intercept_
        return np.expm1(pred) if scale == "count" else pred

    def score(self, df: pd.DataFrame, observed: Optional[str] = None) -> Dict[str, float]:
        """R^2 and mean absolute error, on both the log and the count scale."""
        observed = observed or self.outcome
        data = df[self.used_terms_ + [observed]].replace([np.inf, -np.inf], np.nan).dropna()
        y = data[observed].to_numpy(dtype=float)
        pred = self.predict(data)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return {
            "n": int(len(data)),
            "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "mae_log": float(np.abs(y - pred).mean()),
            "mae_count": float(np.abs(np.expm1(y) - np.expm1(pred)).mean()),
        }

    def residuals(self, df: pd.DataFrame, observed: Optional[str] = None,
                  scale: str = "log") -> np.ndarray:
        observed = observed or self.outcome
        y = df[observed].to_numpy(dtype=float)
        pred = self.predict(df)
        if scale == "count":
            return np.expm1(y) - np.expm1(pred)
        return y - pred

    @property
    def coefficients(self) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted - call fit() first")
        idx = (["intercept"] if self.fit_intercept else []) + list(self.used_terms_)
        vals = ([self.intercept_] if self.fit_intercept else []) + list(self.coef_)
        return pd.Series(vals, index=idx, name="coefficient")

    def summary(self, df: pd.DataFrame):
        """Full statsmodels OLS summary, if statsmodels is available."""
        try:
            import statsmodels.formula.api as smf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "summary() needs statsmodels: pip install 'oif[stats]'"
            ) from exc
        formula = f"{self.outcome} ~ " + " + ".join(self.used_terms_ or self.terms)
        return smf.ols(formula, data=df).fit().summary()

    def __repr__(self) -> str:  # pragma: no cover
        if self.coef_ is None:
            return f"ObjectAttentionModel(terms={list(self.terms)}, unfitted)"
        coefs = ", ".join(f"{k}={v:+.3f}" for k, v in self.coefficients.items())
        return f"ObjectAttentionModel(n={self.n_obs_}, {coefs})"


def cross_validate_by_scene(df: pd.DataFrame, model: Optional[ObjectAttentionModel] = None,
                            scene_col: str = "image", n_folds: int = 5,
                            seed: int = 0) -> pd.DataFrame:
    """Grouped k-fold by scene: fit on some scenes, score on held-out ones.

    Splitting by scene rather than by object is the honest test here. Objects
    within a scene share a viewpoint, a depth range and the same viewers, so
    a random object-level split leaks information across the fold boundary
    and flatters the model.
    """
    model = model or ObjectAttentionModel()
    scenes = df[scene_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    folds = np.array_split(scenes, n_folds)

    rows = []
    for i, held in enumerate(folds):
        train = df[~df[scene_col].isin(held)]
        test = df[df[scene_col].isin(held)]
        if train.empty or test.empty:
            continue
        fitted = ObjectAttentionModel(terms=model.terms, outcome=model.outcome).fit(train)
        stats = fitted.score(test)
        stats.update({"fold": i, "n_train": fitted.n_obs_, "n_scenes_held": len(held)})
        rows.append(stats)
    return pd.DataFrame(rows)
