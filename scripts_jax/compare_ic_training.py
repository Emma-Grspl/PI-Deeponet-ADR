from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generators import get_ic_value as torch_get_ic_value
from src.models.PI_DeepOnet_ADR import PI_DeepONet_ADR
from src_jax.models.pi_deeponet_adr import apply_model, init_model_params
from src_jax.training.step import make_ic_train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n-eval-cases", type=int, default=3)
    parser.add_argument(
        "--torch-config",
        default=str(PROJECT_ROOT / "configs" / "config_ADR_t1_compare.yaml"),
    )
    parser.add_argument(
        "--jax-config",
        default=str(PROJECT_ROOT / "configs_jax" / "config_ADR_jax_t1_compare.yaml"),
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def balanced_type_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    choices = np.array([0, 0, 1, 2, 3, 4], dtype=np.float32)
    return rng.choice(choices, size=(n, 1)).astype(np.float32)


def build_ic_batch_np(rng: np.random.Generator, cfg: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = cfg["physics_ranges"]
    x_min = cfg["geometry"]["x_min"]
    x_max = cfg["geometry"]["x_max"]

    v = rng.uniform(*bounds["v"], size=(n, 1)).astype(np.float32)
    d_val = rng.uniform(*bounds["D"], size=(n, 1)).astype(np.float32)
    mu = rng.uniform(*bounds["mu"], size=(n, 1)).astype(np.float32)
    types = balanced_type_sample(rng, n)
    amplitude = rng.uniform(*bounds["A"], size=(n, 1)).astype(np.float32)
    x0 = rng.uniform(*bounds["x0"], size=(n, 1)).astype(np.float32)
    sigma = rng.uniform(*bounds["sigma"], size=(n, 1)).astype(np.float32)
    k_val = rng.uniform(*bounds["k"], size=(n, 1)).astype(np.float32)

    x_ic = rng.uniform(x_min, x_max, size=(n, 1)).astype(np.float32)
    xt_ic = np.concatenate([x_ic, np.zeros_like(x_ic)], axis=1).astype(np.float32)

    params = np.concatenate([v, d_val, mu, types, amplitude, x0, sigma, k_val], axis=1).astype(np.float32)
    u_true_ic = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        p = {
            "type": float(types[i, 0]),
            "A": float(amplitude[i, 0]),
            "x0": float(x0[i, 0]),
            "sigma": float(sigma[i, 0]),
            "k": float(k_val[i, 0]),
        }
        u_true_ic[i, 0] = float(torch_get_ic_value(x_ic[i, 0], "mixed", p))

    return params, xt_ic, u_true_ic


def make_eval_cases(cfg: dict, seed: int, n_per_family: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    families = {"Tanh": [0], "Sin-Gauss": [1, 2], "Gaussian": [3, 4]}
    cases = []
    for family, type_ids in families.items():
        for _ in range(n_per_family):
            p = {k: rng.uniform(v[0], v[1]) for k, v in cfg["physics_ranges"].items()}
            p["type"] = int(rng.choice(type_ids))
            p["family"] = family
            cases.append(p)
    return cases


def eval_family_curves_torch(model: PI_DeepONet_ADR, cfg: dict, cases: list[dict]) -> dict:
    model.eval()
    x = np.linspace(cfg["geometry"]["x_min"], cfg["geometry"]["x_max"], 400, dtype=np.float32)
    out: dict[str, list[float]] = {}
    with torch.no_grad():
        for case in cases:
            xt = torch.tensor(np.stack([x, np.zeros_like(x)], axis=1), dtype=torch.float32)
            p_vec = np.array(
                [[case["v"], case["D"], case["mu"], case["type"], case["A"], 0.0, case["sigma"], case["k"]]],
                dtype=np.float32,
            )
            p_batch = torch.tensor(np.repeat(p_vec, len(x), axis=0), dtype=torch.float32)
            pred = model(p_batch, xt).cpu().numpy().reshape(-1)
            true = torch_get_ic_value(x, "mixed", case).reshape(-1)
            err = float(np.linalg.norm(true - pred) / (np.linalg.norm(true) + 1e-8))
            out.setdefault(case["family"], []).append(err)
    return {k: float(np.mean(v)) for k, v in out.items()}


def eval_family_curves_jax(params, cfg: dict, cases: list[dict]) -> dict:
    x = np.linspace(cfg["geometry"]["x_min"], cfg["geometry"]["x_max"], 400, dtype=np.float32)
    out: dict[str, list[float]] = {}
    for case in cases:
        xt = np.stack([x, np.zeros_like(x)], axis=1).astype(np.float32)
        p_vec = np.array(
            [[case["v"], case["D"], case["mu"], case["type"], case["A"], 0.0, case["sigma"], case["k"]]],
            dtype=np.float32,
        )
        p_batch = np.repeat(p_vec, len(x), axis=0)
        pred = np.asarray(apply_model(params, jnp.asarray(p_batch), jnp.asarray(xt))).reshape(-1)
        true = torch_get_ic_value(x, "mixed", case).reshape(-1)
        err = float(np.linalg.norm(true - pred) / (np.linalg.norm(true) + 1e-8))
        out.setdefault(case["family"], []).append(err)
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    args = parse_args()
    torch_cfg = load_yaml(args.torch_config)
    jax_cfg = load_yaml(args.jax_config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch_model = PI_DeepONet_ADR(torch_cfg)
    torch_opt = torch.optim.Adam(torch_model.parameters(), lr=args.lr)

    jax_params = init_model_params(jax.random.PRNGKey(args.seed), jax_cfg)
    jax_opt = optax.adam(args.lr)
    jax_opt_state = jax_opt.init(jax_params)
    jax_step = make_ic_train_step(jax_opt)

    rng = np.random.default_rng(args.seed)
    eval_cases = make_eval_cases(torch_cfg, args.seed + 123, args.n_eval_cases)

    history = []
    t0 = time.perf_counter()

    for step_idx in range(1, args.steps + 1):
        params_np, xt_ic_np, u_true_np = build_ic_batch_np(rng, torch_cfg, args.batch_size)

        torch_model.train()
        torch_opt.zero_grad()
        params_t = torch.tensor(params_np, dtype=torch.float32)
        xt_ic_t = torch.tensor(xt_ic_np, dtype=torch.float32)
        u_true_t = torch.tensor(u_true_np, dtype=torch.float32)
        torch_pred = torch_model(params_t, xt_ic_t)
        torch_loss = torch.mean((torch_pred - u_true_t) ** 2)
        torch_loss.backward()
        torch_opt.step()

        jax_batch = (
            jnp.asarray(params_np),
            jnp.zeros((args.batch_size, 2), dtype=jnp.float32),
            jnp.asarray(xt_ic_np),
            jnp.asarray(u_true_np),
            jnp.zeros((args.batch_size, 2), dtype=jnp.float32),
            jnp.zeros((args.batch_size, 2), dtype=jnp.float32),
            jnp.zeros((args.batch_size, 1), dtype=jnp.float32),
            jnp.zeros((args.batch_size, 1), dtype=jnp.float32),
        )
        jax_params, jax_opt_state, jax_loss = jax_step(jax_params, jax_opt_state, jax_batch)

        if step_idx == 1 or step_idx % args.eval_every == 0 or step_idx == args.steps:
            torch_eval = eval_family_curves_torch(torch_model, torch_cfg, eval_cases)
            jax_eval = eval_family_curves_jax(jax_params, jax_cfg, eval_cases)
            snapshot = {
                "step": step_idx,
                "elapsed_sec": time.perf_counter() - t0,
                "torch_loss": float(torch_loss.item()),
                "jax_loss": float(jax.device_get(jax_loss)),
                "torch_eval": torch_eval,
                "jax_eval": jax_eval,
            }
            history.append(snapshot)
            print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
