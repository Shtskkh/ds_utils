"""Single-feature screening for two-class problems.

Quickly rank columns by how well they separate two classes, then describe only
the most promising ones in detail.
"""

from typing import Any, cast

import polars as pl
from scipy import stats
from sklearn.metrics import roc_auc_score

SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "column": pl.String,
    "n_unique": pl.Int64,
    "pct_missing": pl.Float64,
    "auc": pl.Float64,
    "separation_score": pl.Float64,
    "skew": pl.Float64,
    "pct_zero": pl.Float64,
    "ks": pl.Float64,
    "median_diff": pl.Float64,
}


def screen(df: pl.DataFrame, target_col: str) -> pl.DataFrame:
    """Rank columns by single-feature separation of a binary target.

    For every column except ``target_col`` the function computes data-quality
    indicators. For numeric columns with more than two unique values it also
    computes separation, direction and distribution-shape metrics. The result
    is sorted by ``separation_score`` (strongest separators first).

    Parameters
    ----------
    df : polars.DataFrame
        Dataset containing the feature columns and the target column.
    target_col : str
        Name of the binary target column. Must be non-null and contain exactly
        two classes encoded as 0 and 1 (or False/True). Class 1 is treated as
        the "positive" class.

    Returns
    -------
    polars.DataFrame
        One row per feature column, sorted by ``separation_score`` descending.
        Columns without a score (categorical, binary, constant, all-null,
        nested) are placed last. The output always has every column listed
        below, even when no feature qualifies for scoring.

        Columns present for every feature:

        column : str
            Column name.
        n_unique : int | None
            Number of distinct non-missing values. ``1`` means constant; a
            value close to the row count suggests an ID. Both should usually
            be dropped. Small values (2-10) suggest a categorical/ordinal
            column. None for nested (list/struct) columns.
        pct_missing : float
            Share of missing values (0-1). Null, NaN and +/-inf all count as
            missing. High values weaken the AUC because of the median fill
            (see Notes).

        Columns present only for numeric features with more than 2 unique
        values (None otherwise):

        auc : float
            ROC AUC of the feature used as a score for class 1. It is the
            probability that a random class-1 row has a higher value than a
            random class-0 row. 0.5 = no signal; >0.6 weak; >0.7 useful;
            >0.8 strong; <0.5 means class 1 tends to be lower; >0.95 is
            suspicious (possible leakage).
        separation_score : float
            ``abs(auc - 0.5) * 2``. Direction-free separation score in
            [0, 1]; equal to the absolute Cliff's delta. Main ranking metric.
        skew : float
            Skewness of the non-missing values. ``abs(skew) > 1`` suggests
            using median/IQR/percentiles or a log transform instead of
            mean/std.
        pct_zero : float
            Share of exact zeros among non-missing values (0-1). Above ~0.3
            the column likely mixes two behaviours (zero vs non-zero) and
            should be analysed separately for each.
        ks : float | None
            Kolmogorov-Smirnov statistic: maximum gap between the two classes'
            cumulative distributions, in [0, 1]. Unlike ``auc`` it also reacts
            to differences in spread and shape, not only to one-directional
            shifts. None if either class has no observed values.
        median_diff : float | None
            ``median(class 1) - median(class 0)`` in the column's original
            units. The sign gives direction and the magnitude gives practical
            impact. Robust to outliers; not comparable across columns with
            different units. None if either class has no observed values.

    Raises
    ------
    ValueError
        If the target contains nulls or is not exactly the classes 0 and 1.

    Notes
    -----
    * Missing values are filled with the column median **only** for the AUC
      calculation, because ``roc_auc_score`` does not accept None. All other
      class-split metrics use the observed (non-missing) values only. If
      missingness itself differs between classes, the median fill hides that
      signal; check ``pct_missing`` per class separately.
    * AUC captures monotonic separation only. A feature that is both lower and
      higher in class 1 than in class 0 can have ``auc`` near 0.5 and a high
      ``ks``.
    * Features that are weak alone but strong in combination (ratios,
      interactions) will be ranked low.
    * Metrics carry no uncertainty estimate; with small classes bootstrap
      confidence intervals before trusting them. With hundreds of columns,
      some will look good by chance, so validate on held-out data.
    """

    y = df.get_column(target_col)

    if y.dtype == pl.Boolean:
        y = y.cast(pl.Int8)

    if y.null_count() or set(y.unique().to_list()) != {0, 1}:
        raise ValueError("target must be non-null with exactly the classes 0 and 1")

    is_class0 = y.eq(0)
    is_class1 = y.eq(1)
    y_np = y.to_numpy()

    rows: list[dict[str, object]] = []

    for col in df.drop(target_col).columns:
        x = df.get_column(col)
        row: dict[str, object] = {"column": col}

        # Nested columns (list/struct): n_unique may fail or be very slow.
        if x.dtype.is_nested():
            row["n_unique"] = None
            row["pct_missing"] = x.is_null().mean()
            rows.append(row)
            continue

        # Treat NaN and +/-inf as missing so every metric sees the same data.
        if x.dtype.is_float():
            x = x.fill_nan(None)
            x = pl.select(
                pl.when(x.is_infinite()).then(None).otherwise(x).alias(col)
            ).to_series()

        observed = x.drop_nulls()
        n_unique = observed.n_unique() if len(observed) else 0

        row["n_unique"] = n_unique
        row["pct_missing"] = x.is_null().mean()

        if x.dtype.is_numeric() and n_unique > 2:
            # Cast (e.g. Decimal) so sklearn/scipy receive plain floats.
            x = x.cast(pl.Float64)
            median = cast("float", x.median())  # not None: n_unique > 2

            auc = float(roc_auc_score(y_np, x.fill_null(median).to_numpy()))
            row["auc"] = auc
            row["separation_score"] = abs(auc - 0.5) * 2
            row["skew"] = x.skew()
            row["pct_zero"] = x.drop_nulls().eq(0).mean()

            a = x.filter(is_class0).drop_nulls()
            b = x.filter(is_class1).drop_nulls()

            # A class can be entirely missing for this column.
            if len(a) and len(b):
                # scipy's stubs type the result as an opaque placeholder, so
                # basedpyright can't see `.statistic`; it exists at runtime.
                ks_result = cast("Any", stats.ks_2samp(a.to_numpy(), b.to_numpy()))
                row["ks"] = float(ks_result.statistic)

                a_med = cast("float | None", a.median())
                b_med = cast("float | None", b.median())
                row["median_diff"] = (
                    b_med - a_med if a_med is not None and b_med is not None else None
                )

        rows.append(row)

    out = pl.DataFrame(rows, schema=SCHEMA, strict=False)
    return out.sort("separation_score", descending=True, nulls_last=True)
