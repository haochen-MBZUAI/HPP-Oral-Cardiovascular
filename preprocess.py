"""Core preprocessing; inputs are author-selected, same-visit, one-person-one-row tables."""
import numpy as np
import pandas as pd
from phenotypes import RETINAL


def read_table(path):
    path = str(path)
    data = pd.read_feather(path) if path.endswith((".arrow", ".feather")) else pd.read_csv(path)
    data["participant_id"] = data["participant_id"].astype(str)
    return data


def numeric(values):
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def prepare_abundance(raw):
    """>=200 detections -> half-positive-1st-percentile zeros -> sample PPM -> log10.

    PPM uses the row total across the complete microbial layer.
    """
    meta = {"participant_id", "cohort", "research_stage", "array_index", "collection_date", "oral_collection_date", "visit_id"}
    x = raw[[c for c in raw if c not in meta]].apply(numeric)
    if x.isna().any().any() or (x < 0).any().any():
        raise ValueError("Expected finite, nonnegative raw abundances")
    x = x.loc[:, (x > 0).sum() >= 200].astype(float)
    if x.empty:
        raise ValueError("No features passed detection >=200")
    for col in x:
        threshold = x.loc[x[col] > 0, col].quantile(0.01)
        x.loc[x[col] == 0, col] = threshold / 2
    transformed = np.log10(x.div(x.sum(axis=1), axis=0) * 1e6)
    return pd.concat([raw[["participant_id"]], transformed], axis=1)


def clean_phenotypes(raw):
    """Shortest 95% interval; exclude beyond 8 SD; clip remaining values at ±5 SD."""
    result = raw.copy()
    for col in raw.columns.drop("participant_id"):
        values = numeric(raw[col])
        ordered = np.sort(values.dropna())
        if len(ordered) == 0:
            result[col] = values
            continue
        k = max(1, int(np.ceil(0.95 * len(ordered))))
        start = int(np.argmin(ordered[k - 1:] - ordered[:len(ordered) - k + 1]))
        interval = ordered[(ordered >= ordered[start]) & (ordered <= ordered[start + k - 1])]
        mu, sd = interval.mean(), interval.std(ddof=1) if len(interval) > 1 else 0
        result[col] = values.where((values - mu).abs() <= 8 * sd).clip(mu - 5 * sd, mu + 5 * sd)
    return result


def retinal_means(raw_eyes):
    """Demo eye columns use left_<phenotype> and right_<phenotype>."""
    result = raw_eyes[["participant_id"]].copy()
    for p in RETINAL:
        result[p] = (numeric(raw_eyes["left_" + p]) + numeric(raw_eyes["right_" + p])) / 2
    return result


def oral_scores(raw, bleeding="bleeding_gums", abscess="abscess_or_gingivitis", mobility="teeth_mobility_lately"):
    result = raw[["participant_id"]].copy()
    b, a, m = (numeric(raw[c]) for c in (bleeding, abscess, mobility))
    valid = b.isin([0, 1, 2, 3]) & a.isin([0, 1, 2])
    score = np.where((b == 3) & (a == 2), 2, np.where((b == 2) | (a == 2), 1, 0))
    result["soft_tissue_inflammation"] = np.where(valid, score, np.nan)
    result["periodontal_destruction"] = np.where(m.isin([0, 1, 2, 3]), np.where(m == 3, 2, np.where(m == 2, 1, 0)), np.nan)
    return result


def sampling_age(dates, birth_year):
    return pd.to_datetime(dates, errors="coerce").dt.year - numeric(birth_year)
