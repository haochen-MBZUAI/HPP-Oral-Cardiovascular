"""Compact summaries of primary, postural, sensitivity and external associations."""
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from phenotypes import DOMAIN, ORIENTATION, PAIRS, PERSISTENCE_STAGES, MEASURES, REFERENCE_COUNTS


def prioritize(results):
    sig = results[results.significant].copy()
    sig["support"] = -np.log10(sig.adjusted_P.clip(lower=np.nextafter(0., 1.)))
    rank = sig.groupby(["layer", "domain", "feature"]).agg(
        count=("phenotype", "nunique"), support=("support", "sum"), min_P=("adjusted_P", "min")).reset_index()
    rank = rank.sort_values(["layer", "domain", "count", "support", "min_P", "feature"],
                            ascending=[True, True, False, False, True, True])
    selected = rank.groupby(["layer", "domain"]).head(5)
    unique = selected[["layer", "feature"]].drop_duplicates()
    # Join each unique selected feature to its full vascular profile.
    source = results.merge(unique, on=["layer", "feature"], validate="many_to_one")
    return selected, source


def health_profiles(results):
    sig = results[results.significant].copy()
    sig["orientation"] = sig.phenotype.map(ORIENTATION)
    sig["direction"] = np.sign(sig.beta) * sig.orientation
    def label(values):
        signs = set(values[values != 0])
        return "favourable" if signs == {1} else "adverse" if signs == {-1} else "mixed" if {1, -1} <= signs else "unclassified"
    return sig.groupby(["layer", "feature"]).direction.agg(label).rename("descriptive_profile").reset_index()


def landscape(results):
    rows = []
    for (layer, domain), block in results.groupby(["layer", "domain"]):
        sig = block[block.significant]
        reference = REFERENCE_COUNTS[layer] * block.phenotype.nunique()
        rows.append(dict(layer=layer, domain=domain, significant_pairs=len(sig),
                         density_pct=100 * len(sig) / reference, positive=int((sig.beta > 0).sum()), negative=int((sig.beta < 0).sum())))
    return pd.DataFrame(rows)


def bilateral_concordance(primary):
    rows = []
    for layer, all_rows in primary.groupby("layer"):
        for trait, (left, right) in PAIRS.items():
            d = all_rows[all_rows.phenotype.isin([left, right])].pivot(index="feature", columns="phenotype", values=["beta", "significant"])
            beta = d.beta[[left, right]].astype(float).dropna()
            nl, nr = int(d.significant[left].sum()), int(d.significant[right].sum())
            both = d.significant[left] & d.significant[right]
            same = np.sign(d.beta[left].astype(float)) == np.sign(d.beta[right].astype(float))
            rows.append(dict(layer=layer, trait=trait, left=nl, right=nr, bilateral_same_direction=int((both & same).sum()),
                             beta_spearman=spearmanr(beta[left], beta[right]).statistic if len(beta) > 1 else np.nan,
                             imbalance=abs(nl - nr) / (nl + nr) if nl + nr else np.nan))
    return pd.DataFrame(rows)


def postural_patterns(primary):
    rows = []
    for layer, all_rows in primary.groupby("layer"):
        for measure in MEASURES:
            names = [f"{s}_blood_pressure_{measure}" for s in PERSISTENCE_STAGES]
            d = all_rows[all_rows.phenotype.isin(names)].pivot(index="feature", columns="phenotype", values=["beta", "raw_P", "significant"])
            for feature in d.index:
                valid = d.raw_P.loc[feature, names].notna().all() and d.beta.loc[feature, names].notna().all()
                flags = d.significant.loc[feature, names].astype(bool)
                if not flags.any():
                    continue
                pattern = "".join(flags.astype(int).astype(str)) if valid else "incomplete"
                signs = set(np.sign(d.beta.loc[feature, names][flags].astype(float)))
                rows.append(dict(layer=layer, feature=feature, measure=measure, pattern=pattern,
                                 all_stage=pattern == "111", standing_emergent=pattern == "011",
                                 both_standing=pattern in ("111", "011"), same_direction=len(signs) == 1))
    return pd.DataFrame(rows, columns=["layer", "feature", "measure", "pattern", "all_stage", "standing_emergent", "both_standing", "same_direction"])


def sensitivity(primary, adjusted):
    a, b = primary[primary.significant], adjusted[adjusted.significant]
    keys = ["layer", "feature", "phenotype"]
    retained = a.merge(b, on=keys, suffixes=("_primary", "_oral"), validate="one_to_one")
    retained["same_direction"] = np.sign(retained.beta_primary) == np.sign(retained.beta_oral)
    rows = []
    for layer in primary.layer.unique():
        for domain in [*primary.domain.unique(), "Total"]:
            select = lambda x: x[x.layer.eq(layer) & (x.phenotype.map(DOMAIN).eq(domain) if domain != "Total" else True)]
            x, y, both = select(a), select(b), select(retained)
            rows.append(dict(layer=layer, domain=domain, primary=len(x), adjusted=len(y), retained=len(both),
                             retention_pct=100 * len(both) / len(x) if len(x) else np.nan,
                             same_direction_pct=100 * both.same_direction.mean() if len(both) else np.nan))
    return pd.DataFrame(rows)


def genus_candidates(native, strains):
    endpoints = {"lying_blood_pressure_systolic": "sbp", "lying_blood_pressure_pulse_rate": "pulse"}
    rows = []
    for strategy, table in (("native_genus", native), ("strain_to_genus", strains)):
        d = table[table.significant & table.phenotype.isin(endpoints)].copy()
        def genus(feature):
            match = re.search(r"(?:^|\|)g__([^|]+)", feature)
            return match.group(1) if match else re.sub(r"^g__?", "", feature) if strategy == "native_genus" else None
        d["genus"] = d.feature.map(genus)
        if d.genus.isna().any():
            raise ValueError("Provide full genus annotations for significant strain identifiers")
        for (phenotype, g), block in d.groupby(["phenotype", "genus"]):
            signs = set(np.sign(block.beta))
            rows.append(dict(strategy=strategy, genus=g, endpoint=endpoints[phenotype],
                             expected_direction="positive" if signs == {1} else "negative" if signs == {-1} else "mixed"))
    return pd.DataFrame(rows, columns=["strategy", "genus", "endpoint", "expected_direction"])


def compare_nhanes(candidates, external):
    keys = ["genus", "endpoint"]
    external = external.merge(candidates[keys].drop_duplicates(), on=keys, validate="one_to_one")
    valid = external.raw_P.between(0, 1) & np.isfinite(external.beta)
    external["bonferroni_tests"] = int(valid.sum())
    external["adjusted_P"] = (external.raw_P * valid.sum()).clip(upper=1).where(valid)
    d = candidates.merge(external, on=["genus", "endpoint"], how="left", validate="many_to_one")
    d["testable"] = d.raw_P.between(0, 1) & np.isfinite(d.beta)
    d["significant"] = d.testable & d.adjusted_P.lt(0.05)
    d["same_direction"] = (d.expected_direction.eq("positive") & d.beta.gt(0)) | (d.expected_direction.eq("negative") & d.beta.lt(0))
    d["same_direction_significant"] = d.significant & d.same_direction
    return d


def nhanes_replication_summary(comparison):
    rows = []
    for (strategy, endpoint), d in comparison.groupby(["strategy", "endpoint"]):
        n = int(d.testable.sum())
        significant, concordant = int(d.significant.sum()), int(d.same_direction_significant.sum())
        rows.append(dict(strategy=strategy, endpoint=endpoint, testable=n, significant=significant,
                         same_direction_significant=concordant, significant_pct=100 * significant / n if n else np.nan,
                         same_direction_pct=100 * concordant / n if n else np.nan))
    return pd.DataFrame(rows, columns=["strategy", "endpoint", "testable", "significant",
                                       "same_direction_significant", "significant_pct", "same_direction_pct"])
