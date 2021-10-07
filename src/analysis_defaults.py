DEFAULT_DENDRITE_KWARGS = {
    "fluorescence_kwargs": {"norm_agg": "median"},
    "split_unsures": True,
    "high_quality_only": True,
    "search_depth": 7,
    "integrate_over": 5
}

DEFAULT_BIN_KWARGS = {
    "bin_params": 10,
    "bin_per_day": False,
    "equal_bins": True,
    "bin_clip_idx": 10,
}

DEFAULT_MARKOV_KWARGS = {}
DEFAULT_BOOTSTRAP_MARKOV_KWARGS = {"bootstrap": True, "run_exact": True}
DEFAULT_KESTEN_KWARGS = {}