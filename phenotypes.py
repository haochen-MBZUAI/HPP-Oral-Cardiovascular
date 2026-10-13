"""The 34 vascular phenotypes and descriptive orientations in the manuscript."""

STAGES = ("lying", "sitting_second", "standing_one_min", "standing_three_min")
PERSISTENCE_STAGES = ("lying", "standing_one_min", "standing_three_min")
MEASURES = ("systolic", "diastolic", "pulse_rate")
RETINAL = tuple(prefix + name for prefix in ("", "artery_", "vein_")
                for name in ("average_width", "vessel_density", "fractal_dimension", "tortuosity_density"))
PHENOTYPES = {
    "vascular_health": ("from_l_thigh_to_l_ankle_pwv", "from_r_thigh_to_r_ankle_pwv", "l_abi", "r_abi",
                        "l_ankle_pressure", "r_ankle_pressure", "l_brachial_pressure", "r_brachial_pressure"),
    "blood_pressure": tuple(f"{s}_blood_pressure_{m}" for s in STAGES for m in MEASURES),
    "carotid_ultrasound": ("imt_left", "imt_right"),
    "fundus": RETINAL,
}
DOMAIN = {p: domain for domain, names in PHENOTYPES.items() for p in names}
PAIRS = {"ABI": ("l_abi", "r_abi"), "PWV": ("from_l_thigh_to_l_ankle_pwv", "from_r_thigh_to_r_ankle_pwv"),
         "cIMT": ("imt_left", "imt_right"), "ankle_pressure": ("l_ankle_pressure", "r_ankle_pressure"),
         "brachial_pressure": ("l_brachial_pressure", "r_brachial_pressure")}
REFERENCE_COUNTS = {"strain": 783, "gene_family": 355674, "pathway": 786}
# Descriptive Appendix A2 orientation: +1 higher favourable; -1 higher adverse; 0 unclassified.
ORIENTATION = {p: -1 for p in DOMAIN}
for p in ("l_abi", "r_abi", "vessel_density", "fractal_dimension", "artery_average_width",
          "artery_vessel_density", "artery_fractal_dimension", "vein_vessel_density", "vein_fractal_dimension"):
    ORIENTATION[p] = 1
for p in ("l_ankle_pressure", "r_ankle_pressure", "average_width"):
    ORIENTATION[p] = 0
