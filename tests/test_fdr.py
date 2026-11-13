"""BH-FDR families, missing tests, and Python/R agreement."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from models import bh_fdr
from summaries import compare_nhanes


def test_layers_are_separate_but_features_and_outcomes_are_pooled():
    d = pd.DataFrame(dict(layer=["strain"]*3 + ["pathway"]*2,
                          feature=["A", "B", "A", "C", "D"],
                          raw_P=[.01, .04, .03, .001, .8], beta=[1.]*5),
                     index=[9, 9, 3, 5, 2])
    result = bh_fdr(d)
    np.testing.assert_allclose(result.adjusted_P, [.03, .04, .04, .002, .8])
    assert result.fdr_tests.tolist() == [3, 3, 3, 2, 2]
    assert result.index.tolist() == d.index.tolist()
    pd.testing.assert_series_equal(result.raw_P, d.raw_P)


def test_invalid_tests_do_not_enter_family_and_empty_families_work():
    d = pd.DataFrame(dict(layer=["strain"]*7, raw_P=[.01, np.nan, np.inf, -.1, 1.1, .001, .02],
                          beta=[1., 1., 1., 1., 1., np.inf, np.nan]))
    result = bh_fdr(d)
    assert result.fdr_tests.eq(1).all()
    assert result.adjusted_P.iloc[0] == .01
    assert result.adjusted_P.iloc[1:].isna().all()
    assert not result.significant.iloc[1:].any()
    invalid = bh_fdr(d.iloc[1:])
    assert invalid.fdr_tests.eq(0).all() and invalid.adjusted_P.isna().all()
    assert bh_fdr(d.iloc[:0]).empty
    assert bh_fdr(d.iloc[:0], by_layer=False).empty


def test_bh_matches_independent_scipy_implementation():
    rng = np.random.default_rng(43)
    values = np.r_[0., 1., .05, .05, rng.beta(.3, 3, 250)]
    result = bh_fdr(pd.DataFrame(dict(layer="gene_family", beta=1., raw_P=values)))
    np.testing.assert_allclose(result.adjusted_P, false_discovery_control(values, method="bh"))
    boundary = bh_fdr(pd.DataFrame(dict(layer=["strain"], beta=[1.], raw_P=[.05])))
    assert not boundary.significant.iloc[0]


def test_external_family_spans_strategies_and_outcomes_without_duplicates():
    candidates = pd.DataFrame(dict(strategy=["native_genus", "strain_to_genus", "strain_to_genus",
                                             "native_genus", "strain_to_genus"],
                                   genus=["A", "A", "B", "C", "Missing"],
                                   endpoint=["pulse", "pulse", "pulse", "sbp", "sbp"],
                                   expected_direction=["positive"]*5))
    external = pd.DataFrame(dict(genus=["A", "B", "C", "NotACandidate"],
                                 endpoint=["pulse", "pulse", "sbp", "sbp"],
                                 beta=[1.]*4, raw_P=[.01, .04, .03, 1e-20]))
    result = compare_nhanes(candidates, external)
    np.testing.assert_allclose(result.adjusted_P, [.03, .03, .04, .04, np.nan], equal_nan=True)
    assert result.fdr_tests.dropna().eq(3).all()
    assert not result.testable.iloc[-1] and not result.significant.iloc[-1]
    external["adjusted_P"] = 1.
    pd.testing.assert_frame_equal(result, compare_nhanes(candidates, external))


def test_nhanes_r_and_python_use_the_same_bh_family(tmp_path):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript is not installed")
    available = subprocess.run([rscript, "-e", 'quit(status=if(requireNamespace("survey",quietly=TRUE)) 0 else 1)'],
                               capture_output=True)
    if available.returncode:
        pytest.skip("R package survey is not installed")
    rng = np.random.default_rng(25)
    n = 240
    raw = pd.DataFrame(rng.poisson([40, 80, 100], size=(n, 3))+1, columns=["A", "B", "C"])
    raw.insert(0, "sample_id", np.arange(n))
    meta = pd.DataFrame(dict(sample_id=np.arange(n), age=rng.uniform(20, 80, n),
                              sex=np.arange(n)%2, smoking_status=np.array(["Never", "Former", "Current"])[np.arange(n)%3],
                              weight_4yr=rng.uniform(1, 2, n), psu=np.arange(n)%24,
                              strata=(np.arange(n)%24)//6))
    meta["sbp"] = 100 + .2*meta.age + .3*raw.A + rng.normal(0, 7, n)
    meta["pulse"] = 60 + .1*raw.B + rng.normal(0, 7, n)
    candidates = pd.DataFrame(dict(strategy=["native_genus", "strain_to_genus", "native_genus",
                                             "strain_to_genus", "strain_to_genus", "native_genus"],
                                   genus=["A", "A", "B", "A", "B", "Missing"],
                                   endpoint=["sbp", "sbp", "sbp", "pulse", "pulse", "pulse"],
                                   expected_direction=["positive"]*6))
    for name, table in [("raw", raw), ("metadata", meta), ("candidates", candidates)]:
        table.to_csv(tmp_path/f"{name}.csv", index=False)
    script = Path(__file__).resolve().parents[1]/"nhanes_validation.R"
    subprocess.run([rscript, str(script), str(tmp_path/"candidates.csv"), str(tmp_path/"metadata.csv"),
                    str(tmp_path/"raw.csv"), str(tmp_path/"results.csv")], check=True, capture_output=True, text=True)
    external = pd.read_csv(tmp_path/"results.csv")
    valid = np.isfinite(external.raw_P) & np.isfinite(external.beta)
    assert valid.sum() == 4 and external.fdr_tests.eq(4).all()
    np.testing.assert_allclose(external.loc[valid, "adjusted_P"],
                               false_discovery_control(external.loc[valid, "raw_P"], method="bh"))
    comparison = compare_nhanes(candidates, external)
    for _, row in external[valid].iterrows():
        same = comparison.genus.eq(row.genus) & comparison.endpoint.eq(row.endpoint)
        np.testing.assert_allclose(comparison.loc[same, "adjusted_P"], row.adjusted_P)
