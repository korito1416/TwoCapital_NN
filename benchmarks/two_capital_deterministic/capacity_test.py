"""
Capacity vs identification test (deterministic two-capital benchmark, has FD ground truth).

Question: is the ~1e-3 training plateau a NETWORK-CAPACITY limit (the net can't represent
the true solution) or an IDENTIFICATION/optimization limit (the loss can't see the error)?

Test: fit MLPs of increasing size DIRECTLY to the FD ground-truth v(Z), i_d(Z), i_g(Z) by
supervised regression with L-BFGS. L-BFGS removes the SGD optimization difficulty, so the
remaining error is PURE approximation/representation error. If a 32x4 net (the project's
architecture) fits the solution to << 1e-3, then capacity is NOT the bottleneck and the
HJB-training plateau is an identification/conditioning problem (-> preconditioning, not a
bigger net). Caveat: this benchmark is 1-D; the main model is ~6-D, where capacity matters
more -- hence the separate large-net arm in the real retrain.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

import two_capital_model as M
from theta_sensitivity import solve_fd

P = M.load_calibration("A_g_prime_prime")
fine = solve_fd(P, n=4000)
# train grid (what the net sees) and a denser eval grid (true error)
mask = (fine["Z"] >= 0.05) & (fine["Z"] <= 0.95)
Ztr = fine["Z"][mask][::8].reshape(-1, 1)
Zev = fine["Z"][mask][::1].reshape(-1, 1)
targets = {"v": fine["v"][mask], "i_d": fine["i_d"][mask], "i_g": fine["i_g"][mask]}
ytr = {k: v[::8] for k, v in targets.items()}
yev = {k: v[::1] for k, v in targets.items()}

SIZES = [(8, 8), (16, 16), (32,), (32, 32, 32, 32), (64, 64, 64, 64)]


def nparams(layers, din=1, dout=1):
    sizes = [din] + list(layers) + [dout]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


print(f"{'arch':>20} {'params':>7} | {'v RMS':>10} {'v max':>10} | "
      f"{'i_d RMS':>10} {'i_d max':>10} | {'i_g RMS':>10} {'i_g max':>10}")
print("-" * 100)
for layers in SIZES:
    row = {}
    for k in ("v", "i_d", "i_g"):
        net = MLPRegressor(hidden_layer_sizes=layers, activation="tanh", solver="lbfgs",
                           max_iter=20000, tol=1e-12, alpha=1e-9, random_state=0)
        # standardize target for stable lbfgs, invert after
        mu, sd = ytr[k].mean(), ytr[k].std() + 1e-12
        net.fit(Ztr, (ytr[k] - mu) / sd)
        pred = net.predict(Zev) * sd + mu
        err = pred - yev[k]
        row[k] = (np.sqrt(np.mean(err**2)), np.max(np.abs(err)))
    tag = f"{'x'.join(str(n) for n in layers)}"
    print(f"{tag:>20} {nparams(layers):>7} | "
          f"{row['v'][0]:>10.2e} {row['v'][1]:>10.2e} | "
          f"{row['i_d'][0]:>10.2e} {row['i_d'][1]:>10.2e} | "
          f"{row['i_g'][0]:>10.2e} {row['i_g'][1]:>10.2e}")

print("\nInterpretation: if 32x32x32x32 fits v,i_d,i_g to << 1e-3, the 32x4 architecture has")
print("ample CAPACITY to represent the solution -> the ~1e-3 HJB-training plateau is an")
print("IDENTIFICATION/conditioning problem (loss can't see the error), not a capacity limit.")
