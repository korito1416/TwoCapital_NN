"""_smoke_constmap.py — trivial constant economy map, ONLY for smoke-testing
models_terminal_anchor/make_map_anchor.py end-to-end (fit -> checkpoints ->
reload).  Targets: v = 5.0, i_d = 0.04, i_g = 0.10, i_r = 0.005 (pre-tech
regimes; the i_r net must reload as exp(-net) ~ 0.005, verifying the ACTIVE
-log(i_r) convention).  Not an economy — never train from this."""

import numpy as np

MAP_NAME = "smoke_constmap_v5_id004_ig010_ir0005"
PROVENANCE = {
    "purpose": "pipeline smoke test only (make_map_anchor end-to-end)",
    "targets": {"v": 5.0, "i_d": 0.04, "i_g": 0.10, "i_r": 0.005},
}

_POST_TECH = ("PreDamagePostTech", "PostDamagePostTech")


def fields(reg, lk, Z, Y, lr, l3, lx):
    ones = np.ones_like(np.asarray(lk, dtype=float)).reshape(-1, 1)
    out = {"v": 5.0 * ones, "i_d": 0.04 * ones, "i_g": 0.10 * ones}
    out["i_r"] = None if reg in _POST_TECH else 0.005 * ones
    return out
