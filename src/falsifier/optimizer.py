import nevergrad as ng
import numpy as np
import os
import time
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import minimize
from scipy.stats import qmc

def parallel_eval(objective_fn, samples, batch_size=None):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples = len(samples)
    n_jobs = cpu_count()

    if batch_size is None:
        batch_size = max(8, int(np.ceil(n_samples / n_jobs)))

    batches = [samples[i:i + batch_size] for i in range(0, n_samples, batch_size)]

    def evaluate_batch(batch):
        return [objective_fn(s) for s in batch]

    results_nested = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(evaluate_batch)(batch) for batch in batches
    )

    return [val for sublist in results_nested for val in sublist]

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

def optimize_1D(objective_fn, lower_bounds, upper_bounds, num_workers=cpu_count()):
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    assert lower_bounds.shape == upper_bounds.shape

    dim = len(lower_bounds)
    budget = min(2**15, max(4096, int(2**np.ceil(np.log2(100 * dim)))))
    top_k = max(5, int(np.ceil(0.01 * budget)))

    unit_samples = sobol_samples(dim, budget)
    sobol_scaled = lower_bounds + unit_samples * (upper_bounds - lower_bounds)

    objective_values = np.array(parallel_eval(objective_fn, sobol_scaled))
    sorted_indices = np.argsort(objective_values)
    center_point = 0.5 * (lower_bounds + upper_bounds)
    topk_points = sobol_scaled[sorted_indices[:top_k]]
    if objective_fn(center_point) < objective_values[sorted_indices[top_k - 1]]:
        topk_points = np.vstack([topk_points, center_point])

    param = ng.p.Array(shape=(dim,)).set_bounds(lower_bounds, upper_bounds)
    optimizer = ng.optimizers.OnePlusOne(parametrization=param, budget=5000)
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
            options={"gtol": 1e-6, "maxiter": 1000, "eps": 1e-12},
        )
    end_lbfgs = time.time()
    
    return {
        "best_lbfgsb_val": res.fun,
        "best_lbfgsb_x": res.x,
        "total_lbfgs_time": end_lbfgs - start_bfgs,
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
