"""Raw/residual Spearman and bootstrap, plus left/right eye effect concordance."""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from models import join, ols
from phenotypes import DOMAIN, PAIRS, RETINAL, MEASURES, PERSISTENCE_STAGES


def residual_correlation(d, left, right):
    cov = pd.concat([d[["age"]], pd.get_dummies(d[["sex", "smoking"]].astype("category"), drop_first=True, dtype=float)], axis=1)
    cov = sm.add_constant(cov.astype(float), has_constant="add")
    x, y = (sm.OLS(d[c], cov).fit().resid for c in (left, right))
    return spearmanr(x, y).statistic


def bootstrap_correlation(data, left, right, repetitions=1000):
    d = data[["participant_id", left, right, "age", "sex", "smoking"]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(20261013)
    estimates = []
    for _ in range(repetitions):
        sampled = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
        estimates.append((spearmanr(sampled[left], sampled[right]).statistic, residual_correlation(sampled, left, right)))
    raw, adjusted = spearmanr(d[left], d[right]).statistic, residual_correlation(d, left, right)
    ci = np.nanquantile(estimates, [0.025, 0.975], axis=0)
    return dict(N=len(d), raw_rho=raw, adjusted_rho=adjusted,
                raw_CI_lower=ci[0, 0], raw_CI_upper=ci[1, 0], adjusted_CI_lower=ci[0, 1], adjusted_CI_upper=ci[1, 1])


def phenotype_correlations(phenotypes, covariates, eyes=None, repetitions=1000):
    rows = []
    for trait, (left, right) in PAIRS.items():
        d = join(phenotypes[DOMAIN[left]], covariates)
        rows.append(dict(group="vascular", trait=trait, left=left, right=right,
                         **bootstrap_correlation(d, left, right, repetitions)))
    bp = join(phenotypes["blood_pressure"], covariates)
    for measure in MEASURES:
        columns = [f"{s}_blood_pressure_{measure}" for s in PERSISTENCE_STAGES]
        d = bp.dropna(subset=[*columns, "age", "sex", "smoking"])
        for left, right in itertools.combinations(columns, 2):
            rows.append(dict(group="postural", trait=measure, left=left, right=right,
                             **bootstrap_correlation(d, left, right, repetitions)))
    if eyes is not None:
        d = join(eyes, covariates)
        for trait in RETINAL:
            rows.append(dict(group="retinal", trait=trait, left="left_"+trait, right="right_"+trait,
                             **bootstrap_correlation(d, "left_"+trait, "right_"+trait, repetitions)))
    # Each row represents one participant in the same-visit study input.
    return pd.DataFrame(rows)


def eye_effects(abundance, eyes, covariates, layer):
    d = join(join(abundance, eyes), covariates)
    rows = []
    for trait in RETINAL:
        coefficients = [dict(feature=f, left=ols(d, "left_"+trait, f)["beta"], right=ols(d, "right_"+trait, f)["beta"])
                        for f in abundance.columns.drop("participant_id")]
        paired = pd.DataFrame(coefficients).dropna()
        rows.append(dict(layer=layer, trait=trait, paired_features=len(paired),
                         beta_spearman=spearmanr(paired.left, paired.right).statistic if len(paired)>1 else np.nan))
    return pd.DataFrame(rows)
