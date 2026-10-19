"""Ordinary least-squares models and within-layer Benjamini–Hochberg FDR correction."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from phenotypes import DOMAIN, PAIRS
from preprocess import numeric


def join(left, right):
    return left.merge(right, on="participant_id", how="inner", validate="one_to_one")


def ols(data, outcome, feature, extra=()):
    columns = [outcome, feature, "age", "sex", "smoking", *extra]
    d = data[columns].rename(columns={outcome: "y", feature: "x"}).copy()
    for c in ("y", "x", "age", *extra):
        d[c] = numeric(d[c])
    d = d.dropna()
    result = dict(N=len(d), beta=np.nan, raw_P=np.nan, standard_error=np.nan,
                  CI_95_lower=np.nan, CI_95_upper=np.nan)
    if len(d) >= 50 and d.x.nunique() > 1:
        formula = "y ~ x + age + C(sex) + C(smoking)" + "".join(" + " + c for c in extra)
        fit = smf.ols(formula, data=d).fit()
        result.update(beta=fit.params["x"], raw_P=fit.pvalues["x"], standard_error=fit.bse["x"],
                      CI_95_lower=fit.conf_int().loc["x", 0], CI_95_upper=fit.conf_int().loc["x", 1])
    return result


def bh_fdr(results, *, by_layer=True):
    """Correct valid two-sided coefficient P values in each complete test family.

    HPP families are microbial layers; external validation uses one pooled family.
    Call only after assembling all features, outcomes and models in that family.
    """
    result = results.copy()
    valid = (result.raw_P.between(0, 1) & np.isfinite(result.raw_P)
             & np.isfinite(result.beta)).to_numpy()
    adjusted = np.full(len(result), np.nan)
    counts = np.zeros(len(result), dtype=int)
    groups = (result.groupby("layer", sort=False, dropna=False).indices.values()
              if by_layer else [np.arange(len(result))])
    for positions in groups:
        tested = positions[valid[positions]]
        counts[positions] = len(tested)
        if len(tested):
            adjusted[tested] = multipletests(result.raw_P.iloc[tested].to_numpy(),
                                             method="fdr_bh")[1]
    result["adjusted_P"] = adjusted
    result["fdr_tests"] = counts
    result["significant"] = result.adjusted_P < 0.05
    return result


def mwas(abundance, phenotypes, covariates, layer, oral_health=False):
    extra = ("soft_tissue_inflammation", "periodontal_destruction") if oral_health else ()
    rows = []
    for domain, outcomes in phenotypes.items():
        data = join(join(abundance, outcomes), covariates)
        for feature in abundance.columns.drop("participant_id"):
            for outcome in outcomes.columns.drop("participant_id"):
                rows.append(dict(layer=layer, feature=feature, phenotype=outcome, domain=domain,
                                 **ols(data, outcome, feature, extra)))
    return bh_fdr(pd.DataFrame(rows))


def bilateral(abundance, phenotypes, covariates, layer="gene_family"):
    rows = []
    for trait, (left, right) in PAIRS.items():
        data = join(join(abundance, phenotypes[DOMAIN[left]][["participant_id", left, right]]), covariates)
        for feature in abundance.columns.drop("participant_id"):
            d = data[["participant_id", left, right, feature, "age", "sex", "smoking"]].dropna().copy()
            d["bilateral_mean"] = (d[left] + d[right]) / 2
            d["absolute_asymmetry"] = (d[left] - d[right]).abs()
            d["signed_difference"] = d[left] - d[right]
            models = {"left": (left, ()), "right": (right, ()), "mean": ("bilateral_mean", ()),
                      "absolute_asymmetry": ("absolute_asymmetry", ()), "signed_difference": ("signed_difference", ()),
                      "mean_adjusted_absolute_asymmetry": ("absolute_asymmetry", ("bilateral_mean",))}
            for model, (outcome, extra) in models.items():
                rows.append(dict(layer=layer, feature=feature, trait=trait, model=model, **ols(d, outcome, feature, extra)))
    # Joint across all features × five traits × all six models, author-confirmed.
    return bh_fdr(pd.DataFrame(rows))
