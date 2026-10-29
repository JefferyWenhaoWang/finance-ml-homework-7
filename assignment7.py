#!/usr/bin/env python3
"""Assignment 7: reproducible pricing, Greeks, arbitrage penalties and Bonus A.

Run: python assignment7.py --output-dir . --threads 2
All default sizes/epochs match the handout. Evaluation uses CPU float64.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import ndtr
from torch import nn

SEED = 20261028
LOW = np.array([0.5, 0.05, 0.0, 0.05])
HIGH = np.array([1.5, 2.0, 0.05, 0.5])


class MELU(nn.Module):
    """C2 activation in Eq. (2), with safe evaluation of BOTH where branches."""
    def __init__(self, alpha=0.49):
        super().__init__()
        self.alpha = alpha
        self.a = 1 - 2 * alpha
        self.b = 1 / alpha - 2

    def forward(self, z):
        zp = torch.clamp_min(z, 0)
        zn = torch.clamp_max(z, 0)
        positive = (zp.square() / 2 + self.a * zp) / (zp + self.b)
        negative = self.alpha * torch.expm1(zn)
        return torch.where(z > 0, positive, negative)


class PricingNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation_name = activation
        self.register_buffer("low", torch.tensor(LOW, dtype=torch.float32))
        self.register_buffer("width", torch.tensor(HIGH - LOW, dtype=torch.float32))
        act = {"relu": nn.ReLU, "elu": nn.ELU, "melu": MELU}[activation]
        self.layers = nn.Sequential(
            nn.Linear(4, 128), act(), nn.Linear(128, 128), act(),
            nn.Linear(128, 128), act(), nn.Linear(128, 1))

    def forward(self, raw):
        # Inside the graph: all derivatives are w.r.t. RAW m, T, r, sigma.
        z = 2 * (raw - self.low) / self.width - 1
        return self.layers(z).squeeze(-1)


def bs_price(x):
    m, T, r, sigma = x.T
    v = sigma * np.sqrt(T)
    d1 = (np.log(m) + (r + sigma**2 / 2) * T) / v
    return m * ndtr(d1) - np.exp(-r * T) * ndtr(d1 - v)


def data_setup(out):
    rng = np.random.default_rng(SEED)
    x_all = rng.uniform(LOW, HIGH, size=(200_000, 4))
    y_all = bs_price(x_all)
    keep = y_all >= 1e-4
    x, y = x_all[keep], y_all[keep]
    order = rng.permutation(len(x))
    n_train = int(0.8 * len(x))
    itrain, itest = order[:n_train], order[n_train:]
    np.savez_compressed(out / "data_split.npz", x=x, y=y,
                        train_indices=itrain, test_indices=itest)
    # Same OOD points for both penalty models; retain the same price filter.
    ood_low, ood_high = LOW.copy(), HIGH.copy()
    ood_low[1], ood_high[1] = 2, 3
    x_ood = rng.uniform(ood_low, ood_high, size=(50_000, 4))
    x_ood = x_ood[bs_price(x_ood) >= 1e-4]
    np.savez_compressed(out / "bonus_points.npz", x=x_ood)
    meta = {"seed": SEED, "generated": len(x_all), "retained": len(x),
            "dropped": int((~keep).sum()), "drop_percent": float((~keep).mean() * 100),
            "n_train": len(itrain), "n_test": len(itest), "n_bonus": len(x_ood),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "training_dtype": "float32", "evaluation_dtype": "float64",
            "input_order": ["m", "T", "r", "sigma"],
            "input_scaling": "2 * (x - lower_bound) / (upper_bound-lower_bound) - 1"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("DATA", json.dumps(meta), flush=True)
    return x[itrain], y[itrain], x[itest], y[itest], x_ood


def derivatives(model, raw, training=False):
    x = raw.detach().requires_grad_(True)
    c = model(x)
    first = torch.autograd.grad(c.sum(), x, create_graph=True)[0]
    cm, cT = first[:, 0], first[:, 1]
    cmm = torch.autograd.grad(cm.sum(), x, create_graph=training,
                              retain_graph=True)[0][:, 0]
    return x, c, cm, cmm, cT


def activation_checks():
    z = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    value = MELU()(z)
    d1 = torch.autograd.grad(value.sum(), z, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), z)[0]
    checks = {"R_0": value.item(), "Rprime_0": d1.item(), "Rsecond_0": d2.item()}
    for eps in [1e-4, 1e-6, 1e-8]:
        zz = torch.tensor([-eps, eps], dtype=torch.float64, requires_grad=True)
        yy = MELU()(zz)
        dy = torch.autograd.grad(yy.sum(), zz, create_graph=True)[0]
        ddy = torch.autograd.grad(dy.sum(), zz)[0]
        checks[f"Rsecond_at_plusminus_{eps:g}"] = ddy.tolist()
    # Includes the location of the pole in the unused rational branch.
    zz = torch.tensor([-2., -1 / .49 + 2, -.01, 0., .01, 2.],
                      dtype=torch.float64, requires_grad=True)
    yy = MELU()(zz)
    dy = torch.autograd.grad(yy.sum(), zz, create_graph=True)[0]
    ddy = torch.autograd.grad(dy.sum(), zz)[0]
    assert torch.isfinite(yy).all() and torch.isfinite(dy).all() and torch.isfinite(ddy).all()
    assert abs(checks["R_0"]) < 1e-14
    assert abs(checks["Rprime_0"] - .49) < 1e-12
    assert abs(checks["Rsecond_0"] - .49) < 1e-12
    checks["safe_branch_test"] = "passed"
    return checks


def self_checks():
    """Meaningful checks for chain rules, graph flow and precision-sensitive tests."""
    class ExactBS(nn.Module):
        def forward(self, x):
            m, T, r, s = x.unbind(1)
            v = s * torch.sqrt(T)
            d1 = (torch.log(m) + (r + s.square() / 2) * T) / v
            N = lambda q: .5 * (1 + torch.erf(q / math.sqrt(2)))
            return m * N(d1) - torch.exp(-r * T) * N(d1 - v)
    x = torch.tensor([[.8, .5, .02, .2], [1., 1., .03, .3], [1.2, 1.5, .04, .4]],
                     dtype=torch.float64)
    xx, c, cm, cmm, cT = derivatives(ExactBS(), x)
    m, T, r, s = x.unbind(1)
    v = s * torch.sqrt(T)
    d1 = (torch.log(m) + (r + s.square() / 2) * T) / v
    assert torch.allclose(cm, .5 * (1 + torch.erf(d1 / math.sqrt(2))), atol=1e-11)
    assert torch.allclose(cmm, torch.exp(-d1.square()/2) / math.sqrt(2*math.pi) / (m*v), atol=1e-11)
    assert (c - m * cm < 0).all() and (cT > 0).all()
    assert (butterfly_values(ExactBS(), x) >= -1e-14).all()
    # Convex and concave kinks: continuous AD returns zero, discrete test distinguishes them.
    class Kink(nn.Module):
        def __init__(self, sign): super().__init__(); self.sign = sign
        def forward(self, z): return self.sign * torch.relu(z[:, 0] - 1.)
    at_kink = torch.tensor([[1., .5, .02, .2]], dtype=torch.float64)
    assert butterfly_values(Kink(1), at_kink).item() > 0
    assert butterfly_values(Kink(-1), at_kink).item() < 0
    torch.manual_seed(SEED)
    model = PricingNet("melu").double()
    xx, cc, cm, cmm, cT = derivatives(model, x, training=True)
    # Differentiating through c_mm must reach model weights.
    cmm.square().mean().backward()
    assert model.layers[0].weight.grad is not None
    assert torch.isfinite(model.layers[0].weight.grad).all()
    return {"analytic_greeks_and_no_arbitrage": "passed", "discrete_kink_test": "passed",
            "backprop_through_second_derivative": "passed", **activation_checks()}


def train(model, xtrain, ytrain, out, name, *, epochs, batch_size, lr, penalty=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(len(xtrain) / batch_size)
    scheduler = None if penalty is not None else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch, eta_min=0)
    # Reset for every run: same mini-batch order, independent of activation or lambda.
    rng = np.random.default_rng(SEED + 1)
    x = torch.tensor(xtrain, dtype=torch.float32)
    y = torch.tensor(ytrain, dtype=torch.float32)
    start = time.perf_counter()
    history = []
    model.train()
    for epoch in range(epochs):
        order = torch.from_numpy(rng.permutation(len(x)))
        mse_sum, p_sum = 0., 0.
        for idx in order.split(batch_size):
            xb, yb = x[idx], y[idx]
            optimizer.zero_grad(set_to_none=True)
            if penalty is not None and penalty[0] != 0:
                lam, p = penalty
                raw, pred, cm, cmm, cT = derivatives(model, xb, training=True)
                violations = torch.stack((-raw[:,0].square()*cmm, -raw[:,1]*cT,
                                          pred-raw[:,0]*cm, -pred), dim=1)
                P = torch.relu(violations).pow(p).sum(dim=1).mean()
            else:
                pred = model(xb)
                P = pred.new_zeros(())
            mse = (pred - yb).square().mean()
            loss = mse + (penalty[0] * P if penalty is not None else 0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {name}, epoch {epoch+1}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            mse_sum += mse.item() * len(idx)
            p_sum += P.item() * len(idx)
        record = {"epoch": epoch+1, "train_mse": mse_sum / len(x),
                  "penalty": p_sum / len(x), "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": time.perf_counter() - start}
        history.append(record)
        print(name, json.dumps(record), flush=True)
        torch.save({"state_dict": model.state_dict(), "activation": model.activation_name,
                    "epoch": epoch+1, "seed": SEED}, out / "models" / f"{name}.pt")
        (out / "results" / f"history_{name}.json").write_text(json.dumps(history, indent=2))
    return model


def load_model(out, name):
    checkpoint = torch.load(out / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
    model = PricingNet(checkpoint["activation"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.double().eval()


@torch.no_grad()
def predict(model, x, batch_size=2048):
    xt = torch.as_tensor(x, dtype=next(model.parameters()).dtype)
    return torch.cat([model(b) for b in xt.split(batch_size)]).numpy()


@torch.no_grad()
def butterfly_values(model, x):
    # Divide the dollar butterfly by positive K2; its sign is unchanged.
    # Fixed S, K1=.98*K2, K3=1.02*K2 => m1=m/.98, m3=m/1.02.
    xm, xp = x.clone(), x.clone()
    xm[:, 0] /= .98
    xp[:, 0] /= 1.02
    return .98 * model(xm) - 2 * model(x) + 1.02 * model(xp)


def metrics(model, x, y=None):
    counts = np.zeros(5, dtype=np.int64)
    counts_tol = np.zeros(5, dtype=np.int64)
    min_residual = np.full(5, np.inf)
    max_violation = np.zeros(5)
    for raw in torch.as_tensor(x, dtype=torch.float64).split(1024):
        xx, c, cm, cmm, cT = derivatives(model, raw)
        # All residuals should be NONNEGATIVE. Test the exact signs requested.
        residuals = torch.stack((xx[:,0]*cm-c, cmm, cT, c,
                                  butterfly_values(model, raw)), dim=1).detach().numpy()
        assert np.isfinite(residuals).all()
        counts += (residuals < 0).sum(0)
        counts_tol += (residuals < -1e-12).sum(0)
        min_residual = np.minimum(min_residual, residuals.min(0))
        max_violation = np.maximum(max_violation, np.maximum(-residuals, 0).max(0))
    names = ["strike", "convexity", "maturity", "nonnegative", "butterfly"]
    result = {"n": len(x)}
    for i, key in enumerate(names):
        result[f"{key}_pct"] = float(counts[i] / len(x) * 100)
        result[f"{key}_count"] = int(counts[i])
        result[f"{key}_pct_tol_1e-12"] = float(counts_tol[i] / len(x) * 100)
        result[f"{key}_min_residual"] = float(min_residual[i])
        result[f"{key}_max_violation"] = float(max_violation[i])
    if y is not None:
        result["price_rmse_bp"] = float(1e4 * np.sqrt(np.mean((predict(model, x) - y)**2)))
    return result


def greek_slice(model):
    m = np.linspace(.6, 1.4, 801)
    x = np.column_stack((m, np.full_like(m,.5), np.full_like(m,.02), np.full_like(m,.2)))
    xx, c, cm, cmm, cT = derivatives(model, torch.tensor(x, dtype=torch.float64))
    d1 = (np.log(m)+(.02+.2**2/2)*.5)/(.2*np.sqrt(.5))
    delta = ndtr(d1)
    gamma = np.exp(-d1**2/2) / np.sqrt(2*np.pi) / (m*.2*np.sqrt(.5))
    dm, gm = cm.detach().numpy(), cmm.detach().numpy()
    return {"m": m, "delta_exact": delta, "scaled_gamma_exact": gamma,
            "delta": dm, "scaled_gamma": gm,
            "delta_relative_L2": float(np.linalg.norm(dm-delta)/np.linalg.norm(delta)),
            "gamma_relative_L2": float(np.linalg.norm(gm-gamma)/np.linalg.norm(gamma))}


def save_csv(path, rows):
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(out, xtest, ytest, xood):
    results = {"base": {}, "penalties": {}, "bonus": {}, "checks": self_checks()}
    colors = {"relu":"#D55E00", "elu":"#0072B2", "melu":"#009E73"}
    greek_data = {}
    for name in ["relu", "elu", "melu"]:
        model = load_model(out, name)
        result = metrics(model, xtest, ytest)
        gd = greek_slice(model)
        result.update({k:gd[k] for k in ["delta_relative_L2", "gamma_relative_L2"]})
        results["base"][name] = result
        greek_data[name] = gd
        np.savez_compressed(out / "results" / f"slice_{name}.npz", **gd)
        print("EVAL", name, json.dumps(result), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout="constrained")
    for ax, q, exact in zip(axes, ["delta", "scaled_gamma"], ["delta_exact", "scaled_gamma_exact"]):
        ax.plot(gd["m"], gd[exact], "k--", lw=2, label="Black-Scholes")
        for name in colors:
            ax.plot(greek_data[name]["m"], greek_data[name][q], color=colors[name], label=name.upper(), lw=1.4)
        ax.set(xlabel="Moneyness m = S/K", ylabel="Delta" if q=="delta" else "Scaled gamma K Gamma")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("T=0.5, r=0.02, sigma=0.2 | automatic differentiation w.r.t. raw inputs")
    fig.savefig(out/"figures"/"greeks_comparison.png", dpi=180)
    plt.close(fig)
    # Enlarged Gamma panel so that ELU's nonsmooth second derivative is visible.
    fig, axes = plt.subplots(2,1,figsize=(9,6),sharex=True,layout="constrained")
    for name in ["elu", "melu"]:
        axes[0].plot(gd["m"], greek_data[name]["scaled_gamma"], label=name.upper(), color=colors[name], lw=1)
        axes[1].plot(gd["m"], greek_data[name]["scaled_gamma"]-gd["scaled_gamma_exact"],label=name.upper(),color=colors[name],lw=1)
    axes[0].plot(gd["m"],gd["scaled_gamma_exact"],"k--",label="Black-Scholes")
    axes[0].set(ylabel="K Gamma");axes[1].set(xlabel="Moneyness m",ylabel="K Gamma error")
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/"figures"/"gamma_detail.png",dpi=180);plt.close(fig)
    for lam, p in [(0,2),(1,2),(10,2),(100,2),(100,4)]:
        name=f"melu_l{lam}_p{p}"
        model=load_model(out,name)
        results["penalties"][name] = {"lambda":lam,"p":p,**metrics(model,xtest,ytest)}
        if (lam,p) in [(0,2),(100,2)]:
            results["bonus"][name] = {"lambda":lam,"p":p,**metrics(model,xood,bs_price(xood))}
        print("EVAL",name,json.dumps(results["penalties"][name]),flush=True)
    # A theoretical exact-BS reference catches incorrect signs and strike rescaling.
    from scipy.special import ndtr as normal_cdf
    d = (np.log(xtest[:,0])+(xtest[:,2]+xtest[:,3]**2/2)*xtest[:,1])/(xtest[:,3]*np.sqrt(xtest[:,1]))
    results["checks"]["exact_BS_strike_residual_min"] = float(np.min(
        np.exp(-xtest[:,2]*xtest[:,1])*normal_cdf(d-xtest[:,3]*np.sqrt(xtest[:,1]))))
    (out/"results"/"all_results.json").write_text(json.dumps(results,indent=2))
    save_csv(out/"results"/"base_metrics.csv",[{"activation":n,**r} for n,r in results["base"].items()])
    save_csv(out/"results"/"penalty_metrics.csv",[{"model":n,**r} for n,r in results["penalties"].items()])
    save_csv(out/"results"/"bonus_metrics.csv",[{"model":n,**r} for n,r in results["bonus"].items()])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for name,color in colors.items():
        h=json.loads((out/"results"/f"history_{name}.json").read_text())
        axes[0].semilogy([v["epoch"] for v in h],[1e4*math.sqrt(v["train_mse"]) for v in h],label=name.upper(),color=color)
    axes[0].set(xlabel="Epoch",ylabel="Training RMSE (bp of strike)");axes[0].legend();axes[0].grid(alpha=.25)
    rows=list(results["penalties"].values())
    labels=[f"L={r['lambda']}, p={r['p']}" for r in rows]
    axes[1].bar(np.arange(len(rows)),[r["price_rmse_bp"] for r in rows],color="#0072B2")
    axes[1].set(xticks=np.arange(len(rows)),xticklabels=labels,ylabel="Test RMSE (bp of strike)")
    axes[1].tick_params(axis="x",rotation=30)
    fig.savefig(out/"figures"/"training_and_penalty_accuracy.png",dpi=180);plt.close(fig)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--stage",choices=["all","train","evaluate","check"],default="all")
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Deterministic CPU operations; no batch normalization or dropout.
    torch.use_deterministic_algorithms(True)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    for d in ["models","results","figures"]:(out/d).mkdir(exist_ok=True)
    checks=self_checks();(out/"results"/"self_checks.json").write_text(json.dumps(checks,indent=2))
    if args.stage=="check":print(json.dumps(checks,indent=2));return
    if args.stage in ["all","train"]:
        xtrain,ytrain,xtest,ytest,xood=data_setup(out)
        for activation in ["relu","elu","melu"]:
            torch.manual_seed(SEED)
            model=PricingNet(activation)
            train(model,xtrain,ytrain,out,activation,epochs=40,batch_size=128,lr=2e-3)
        base=load_model(out,"melu").float()
        for lam,p in [(0,2),(1,2),(10,2),(100,2),(100,4)]:
            model=copy.deepcopy(base)
            train(model,xtrain,ytrain,out,f"melu_l{lam}_p{p}",epochs=10,batch_size=256,lr=5e-4,penalty=(lam,p))
    else:
        z=np.load(out/"data_split.npz");xtest=z["x"][z["test_indices"]];ytest=z["y"][z["test_indices"]]
        xood=np.load(out/"bonus_points.npz")["x"]
    if args.stage in ["all","evaluate"]:evaluate(out,xtest,ytest,xood)
    print("DONE",flush=True)


if __name__=="__main__":main()
