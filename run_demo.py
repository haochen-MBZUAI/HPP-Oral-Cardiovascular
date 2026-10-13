"""Run the manuscript analysis logic on study tables in data/."""
from pathlib import Path
import sys
import pandas as pd
from phenotypes import PHENOTYPES, RETINAL
from preprocess import read_table, prepare_abundance, clean_phenotypes, retinal_means, oral_scores, sampling_age
from models import join, mwas, bilateral
from summaries import prioritize, landscape, health_profiles, bilateral_concordance, postural_patterns, sensitivity, genus_candidates
from paired_correlations import phenotype_correlations, eye_effects


def run(data_dir="data", output_dir="outputs", bootstrap=1000):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save = lambda table, name: table.to_csv(output_dir / name, index=False)
    cov = read_table(data_dir / "covariates.csv")
    if {"oral_collection_date", "year_of_birth"} <= set(cov):
        cov["age"] = sampling_age(cov.oral_collection_date, cov.year_of_birth)
    cov = cov[["participant_id", "age", "sex", "smoking"]]
    raw = {domain: read_table(data_dir / f"{domain}.csv") for domain in PHENOTYPES}
    eyes = None
    if (data_dir / "paired_eyes.csv").exists():
        raw_eyes = read_table(data_dir / "paired_eyes.csv")
        raw["fundus"] = retinal_means(raw_eyes)
        columns = [side + p for side in ("left_", "right_") for p in RETINAL]
        eyes = clean_phenotypes(raw_eyes[["participant_id", *columns]])
    phenotypes = {domain: clean_phenotypes(table[["participant_id", *PHENOTYPES[domain]]]) for domain, table in raw.items()}
    primary, abundance = {}, {}
    for layer in ("strain", "gene_family", "pathway", "genus"):
        source = data_dir / f"{layer}.csv"
        if layer == "gene_family" and (data_dir / "gene_family.arrow").exists():
            source = data_dir / "gene_family.arrow"
        abundance[layer] = prepare_abundance(read_table(source))
        primary[layer] = mwas(abundance[layer], phenotypes, cov, layer)
        save(primary[layer], f"{layer}_primary.csv")
    main = pd.concat([primary[layer] for layer in ("strain", "gene_family", "pathway")], ignore_index=True)
    save(landscape(main), "association_landscape.csv")
    save(bilateral_concordance(main), "primary_bilateral_concordance.csv")
    save(postural_patterns(main), "postural_patterns.csv")
    selected, heatmap = prioritize(main)
    save(selected, "prioritized_features.csv")
    save(heatmap, "heatmap_source.csv")
    save(health_profiles(heatmap), "descriptive_health_profiles.csv")
    save(bilateral(abundance["gene_family"], phenotypes, cov), "gene_family_bilateral.csv")
    oral_cov = join(cov, oral_scores(read_table(data_dir / "oral_health.csv")))
    for layer in ("strain", "gene_family", "pathway"):
        adjusted = mwas(abundance[layer], phenotypes, oral_cov, layer, oral_health=True)
        save(adjusted, f"{layer}_oral_health.csv")
        save(sensitivity(primary[layer], adjusted), f"{layer}_sensitivity.csv")
        if eyes is not None:
            save(eye_effects(abundance[layer], eyes, cov, layer), f"{layer}_eye_effect_concordance.csv")
    save(phenotype_correlations(phenotypes, cov, eyes, bootstrap), "paired_correlations.csv")
    save(genus_candidates(primary["genus"], primary["strain"]), "NHANES_candidates.csv")
    rows = []
    for domain, table in phenotypes.items():
        d = join(table, cov)
        for sex, group in (("All", d), ("Male", d[d.sex == 1]), ("Female", d[d.sex == 0])):
            for p in PHENOTYPES[domain]:
                x = group[p].dropna()
                rows.append(dict(domain=domain, phenotype=p, sex=sex, N=len(x), mean=x.mean(), sd=x.std(ddof=1)))
    save(pd.DataFrame(rows), "phenotype_summary_by_sex.csv")
    print(f"Analysis outputs: {output_dir}")


if __name__ == "__main__":
    run(*sys.argv[1:3])
