import numpy as np
import time
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import minimize
from scipy.stats import qmc

def parallel_eval(objective_fn, samples):
    samples = np.asarray(samples, dtype=np.float32)
    n_jobs = min(cpu_count(), len(samples))

    def eval_single(x): return objective_fn(x)
    
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(eval_single)(x) for x in samples
    )
    return results

def sobol_samples(dim, n_samples):
    sampler = qmc.Sobol(dim, scramble=False)
    return sampler.random(n_samples)

def optimize_1D(objective_fn, lower_bounds, upper_bounds):
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    dim = len(lower_bounds)
    budget = min(2**15, max(4096, int(2**np.ceil(np.log2(100 * dim)))))
    top_k = 3

    unit_samples = sobol_samples(dim, budget)
    sobol_scaled = lower_bounds + unit_samples * (upper_bounds - lower_bounds)

    objective_values = np.array(parallel_eval(objective_fn, sobol_scaled))
    sorted_indices = np.argsort(objective_values)
    center_point = 0.5 * (lower_bounds + upper_bounds)
    topk_points = sobol_scaled[sorted_indices[:top_k]]

    if objective_fn(center_point) < objective_values[sorted_indices[top_k - 1]]:
        topk_points = np.vstack([topk_points, center_point])

    best_val = float("inf")
    best_x = None
    total_time = 0.0

    for init_point in topk_points:
        start_time = time.time()
        res = minimize(
            objective_fn,
            init_point,
            method="L-BFGS-B",
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={"gtol": 1e-6, "maxiter": 1000, "eps": 1e-12},
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    return {
        "best_lbfgsb_val": best_val,
        "best_lbfgsb_x": best_x,
        "total_lbfgs_time": total_time,
    }
