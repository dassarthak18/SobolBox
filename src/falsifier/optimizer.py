import nevergrad as ng
import numpy as np
import os
import time
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import minimize
from scipy.stats import qmc

def sobol_samples(dim, n_samples, cache_dir=".sobol_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sobol_d{dim}_n{n_samples}.npy")

    if os.path.isfile(cache_path):
        unit_samples = np.load(cache_path)
    else:
        sampler = qmc.Sobol(dim, scramble=False)
        unit_samples = sampler.random(n_samples)
        np.save(cache_path, unit_samples)

    return unit_samples

def optimize_1D(objective_fn, lower_bounds, upper_bounds, topk_points, eps=1e-12):
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    assert lower_bounds.shape == upper_bounds.shape

    mask = (lower_bounds == upper_bounds)
    eps_lower = np.where(mask, lower_bounds - eps, lower_bounds)
    eps_upper = np.where(mask, upper_bounds + eps, upper_bounds)

    dim = len(lower_bounds)
    center_point = 0.5 * (lower_bounds + upper_bounds)

    param = ng.p.Array(shape=(dim,))
    span = np.asarray(eps_upper) - np.asarray(eps_lower)
    sigma = np.maximum(span / 6.0, 1e-12)
    param.set_mutation(sigma=sigma)
    param.value = center_point.copy()
    param.set_bounds(eps_lower, eps_upper)
    optimizer = ng.optimizers.OnePlusOne(parametrization=param, budget=min(5000, max(1000, 100 * dim)))
    for x0 in topk_points:
        candidate = optimizer.parametrization.spawn_child()
        candidate.value = x0
        optimizer.tell(candidate, objective_fn(x0))

    while optimizer.num_tell < optimizer.budget:
        candidate = optimizer.ask()
        value = objective_fn(candidate.value)
        optimizer.tell(candidate, value)

    start_lbfgs = time.time()
    recommendation = optimizer.provide_recommendation()
    res = minimize(
            objective_fn,
            recommendation.value,
            method="L-BFGS-B",
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={"gtol": 1e-12, "maxiter": 10000, "eps": 1e-12},
        )
    end_lbfgs = time.time()
    
    return {
        "best_lbfgsb_val": res.fun,
        "best_lbfgsb_x": res.x,
        "total_lbfgs_time": end_lbfgs - start_lbfgs,
    }

    '''
    best_lbfgs_val = float("inf")
    best_lbfgs_x = None
    total_lbfgs_time = 0.0

    for i, init_point in enumerate(topk_points):
        start_lbfgs = time.time()
        res = minimize(
            objective_fn,
            init_point,
            method="L-BFGS-B",
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={"gtol": 1e-6, "maxiter": 1000, "eps": 1e-12},
        )
        lbfgs_time = time.time() - start_lbfgs
        total_lbfgs_time += lbfgs_time
        val_lbfgs = res.fun

        if val_lbfgs < best_lbfgs_val:
            best_lbfgs_val = val_lbfgs
            best_lbfgs_x = res.x

    return {
        "best_lbfgsb_val": best_lbfgs_val,
        "best_lbfgsb_x": best_lbfgs_x,
        "total_lbfgs_time": total_lbfgs_time,
    }
    '''
