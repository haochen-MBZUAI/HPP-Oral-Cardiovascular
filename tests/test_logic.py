import numpy as np
import pandas as pd
from models import bonferroni, bilateral, mwas
from preprocess import prepare_abundance, clean_phenotypes, oral_scores
from phenotypes import PHENOTYPES
from summaries import prioritize, compare_nhanes, bilateral_concordance, postural_patterns, nhanes_replication_summary


def test_global_correction():
    d = pd.DataFrame(dict(layer=["strain"]*4, feature=["A","A","B","B"], beta=[1]*4, raw_P=[.01,.02,.01,np.nan]))
    result = bonferroni(d)
    np.testing.assert_allclose(result.adjusted_P,[.03,.06,.03,np.nan],equal_nan=True)


def test_preprocessing():
    raw = pd.DataFrame(dict(participant_id=range(210),A=[0]+[2]*209,B=[4]*210,rare=[1]*199+[0]*11))
    d = prepare_abundance(raw)
    assert "rare" not in d
    assert d.A.iloc[0] == np.log10(1/5*1e6)
    np.testing.assert_allclose(10**d.A+10**d.B,1e6)
    clean = clean_phenotypes(pd.DataFrame(dict(participant_id=range(100),x=[0]*96+[100,200,300,400])))
    assert clean.x.isna().sum()==4
    boundary = pd.DataFrame(dict(participant_id=range(210),exactly_200=[1]*200+[0]*10,B=[4]*210))
    assert "exactly_200" in prepare_abundance(boundary)


def test_missing_oral_answers():
    d = pd.DataFrame(dict(participant_id=range(3),bleeding_gums=[3,2,np.nan],abscess_or_gingivitis=[2,0,0],teeth_mobility_lately=[3,2,np.nan]))
    result = oral_scores(d)
    np.testing.assert_allclose(result.soft_tissue_inflammation,[2,1,np.nan],equal_nan=True)
    np.testing.assert_allclose(result.periodontal_destruction,[2,1,np.nan],equal_nan=True)


def test_primary_and_paired_models():
    rng=np.random.default_rng(6)
    n=210
    a=prepare_abundance(pd.DataFrame(dict(participant_id=range(n),A=rng.uniform(1,3,n),B=rng.uniform(1,3,n))))
    cov=pd.DataFrame(dict(participant_id=range(n),age=rng.uniform(30,70,n),sex=np.arange(n)%2,smoking=np.arange(n)%3))
    phenotypes={domain:pd.DataFrame({"participant_id":range(n),**{p:2*a.A+rng.normal(size=n) for p in names}}) for domain,names in PHENOTYPES.items()}
    phenotypes["carotid_ultrasound"].loc[:3,"imt_right"]=np.nan
    primary=mwas(a,phenotypes,cov,"gene_family")
    assert len(primary)==68 and primary.bonferroni_tests.eq(68).all()
    concordance=bilateral_concordance(primary)
    assert len(concordance)==5 and concordance.beta_spearman.abs().le(1).all()
    assert "both_standing" in postural_patterns(primary)
    pairs=bilateral(a,phenotypes,cov)
    assert len(pairs)==60 and pairs.bonferroni_tests.eq(60).all()
    for _,block in pairs.groupby(["feature","trait"]):
        assert block.N.nunique()==1
        beta=block.set_index("model").beta
        np.testing.assert_allclose(beta["mean"],(beta["left"]+beta["right"])/2,atol=1e-12)
        np.testing.assert_allclose(beta["signed_difference"],beta["left"]-beta["right"],atol=1e-12)


def test_ranking_support_and_deduplication():
    d=pd.DataFrame(dict(layer=["strain"]*4,domain=["vascular_health"]*4,feature=["A","A","B","B"],phenotype=["l_abi","r_abi"]*2,
                        beta=[1]*4,adjusted_P=[1e-12,.04,1e-8,1e-8],significant=[True]*4))
    selected,source=prioritize(d)
    assert selected.feature.tolist()==["B","A"]
    assert len(source)==4


def test_mixed_hpp_signs_do_not_count_as_concordant():
    candidates=pd.DataFrame(dict(strategy=["strain_to_genus"],genus=["Prevotella"],endpoint=["pulse"],expected_direction=["mixed"]))
    external=pd.DataFrame(dict(genus=["Prevotella"],endpoint=["pulse"],beta=[1.],raw_P=[.049]))
    d=compare_nhanes(candidates,external)
    assert d.significant.iloc[0] and not d.same_direction_significant.iloc[0]


def test_external_bonferroni_pools_unique_tests_across_strategies():
    candidates=pd.DataFrame(dict(strategy=["native_genus","strain_to_genus","strain_to_genus"],
        genus=["A","A","B"],endpoint=["pulse"]*3,expected_direction=["positive"]*3))
    external=pd.DataFrame(dict(genus=["A","B"],endpoint=["pulse"]*2,beta=[1.,1.],raw_P=[.02,.03]))
    d=compare_nhanes(candidates,external)
    assert d.bonferroni_tests.eq(2).all()
    np.testing.assert_allclose(d.adjusted_P,[.04,.04,.06])
    assert d.significant.tolist()==[True,True,False]
    summary=nhanes_replication_summary(d).set_index("strategy")
    assert summary.loc["strain_to_genus","significant"]==1
    assert summary.loc["strain_to_genus","testable"]==2
