"""Step 4 - the same problem with BoTorch: qNEHVI (batch q=2).

Independent GPs per objective (SingleTaskGP in a ModelListGP), objectives
negated because BoTorch maximises, reference point from the problem.
Compares the hypervolume trace with the from-scratch ParEGO of step 3.

Requires:  pip install torch botorch
"""
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

warnings.filterwarnings("ignore")
torch.manual_seed(0)
tkw = dict(dtype=torch.double)
problem = BraninCurrin(negate=True).to(**tkw)          # maximise -f
bounds = problem.bounds
REF = problem.ref_point                                # (-18, -6)
N_INIT, N_BATCH, Q = 6, 15, 2


def fit_model(X, Y):
    models = [SingleTaskGP(X, Y[:, i:i + 1], input_transform=Normalize(2),
                           outcome_transform=Standardize(1)) for i in range(Y.shape[1])]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def hv(Y):
    return Hypervolume(REF).compute(Y[is_non_dominated(Y)])


def run(seed):
    torch.manual_seed(seed)
    X = torch.rand(N_INIT, 2, **tkw) * (bounds[1] - bounds[0]) + bounds[0]
    Y = problem(X)
    trace = [hv(Y)]
    for _ in range(N_BATCH):
        model = fit_model(X, Y)
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=REF, X_baseline=X, prune_baseline=True,
            sampler=SobolQMCNormalSampler(torch.Size([128])))
        cand, _ = optimize_acqf(acq, bounds=bounds, q=Q, num_restarts=10, raw_samples=256)
        X = torch.cat([X, cand])
        Y = torch.cat([Y, problem(cand)])
        trace.append(hv(Y))
    return X, Y, trace


if __name__ == "__main__":
    traces = []
    for s in range(3):
        X, Y, tr = run(s)
        traces.append(tr)
        print(f"seed {s}: final hypervolume = {tr[-1]:.2f}  ({len(X)} evaluations)")
    print(f"max achievable hypervolume for Branin-Currin ~ {problem.max_hv:.2f}")
    T = np.array(traces)
    evals = N_INIT + Q * np.arange(N_BATCH + 1)
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(evals, T.mean(0), "o-", ms=3, label="qNEHVI (BoTorch)")
    plt.fill_between(evals, T.min(0), T.max(0), alpha=0.2)
    plt.axhline(problem.max_hv, color="k", ls="--", lw=1, label="max HV")
    plt.xlabel("evaluations")
    plt.ylabel("hypervolume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/04_qnehvi.png", dpi=130)
