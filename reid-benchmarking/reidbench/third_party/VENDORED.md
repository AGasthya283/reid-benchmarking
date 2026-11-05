# Vendored third-party code

| Path | Upstream | Commit | License | Changes |
|---|---|---|---|---|
| `trackeval/` | https://github.com/JonathonLuiten/TrackEval | 12c8791 | MIT | removed `baselines/`, `plotting` and all dataset loaders except MOTChallenge 2D box; NumPy 2 fixes (`np.float/int/bool` -> builtins) |

Trackers are vendored under `reidbench/trackers/` (see the top-level README, "Tracker changes").
