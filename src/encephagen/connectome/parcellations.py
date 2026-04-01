"""Brain parcellation definitions and region type classifications."""

# Classify region labels into functional types based on name patterns.
# Used by the registry to assign specialized modules to regions.

REGION_TYPE_PATTERNS: dict[str, list[str]] = {
    "thalamus": ["thalamus", "thalam", "lgn", "mgn", "pulvinar"],
    "hippocampus": ["hippocampus", "hippocamp", "dentate", "ca1", "ca3", "subiculum"],
    "amygdala": ["amygdala"],
    "cerebellum": ["cerebellum", "cerebell"],
    "basal_ganglia": ["caudate", "putamen", "pallidum", "accumbens", "striatum"],
    "brainstem": ["brainstem", "pons", "medulla", "midbrain"],
}


def classify_region(label: str) -> str:
    """Classify a region label into a functional type. Returns 'cortical' by default."""
    label_lower = label.lower()
    for region_type, patterns in REGION_TYPE_PATTERNS.items():
        if any(p in label_lower for p in patterns):
            return region_type
    return "cortical"
