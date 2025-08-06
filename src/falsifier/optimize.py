import numpy as np
import time, os
from joblib import Parallel, delayed, cpu_count
from scipy.optimize import minimize
from scipy.stats import qmc

def black_box_batch(sess, input_array_batch, input_name, label_name, input_shape):
    batch_size = len(input_array_batch)
    reshaped_batch = np.asarray(input_array_batch, dtype=np.float32).reshape((batch_size,) + tuple(input_shape))

    try:
        outputs = sess.run([label_name], {input_name: reshaped_batch})[0]
    except TypeError:
        outputs = sess.run([label_name], {input_name: reshaped_batch})[0]

    return outputs.tolist()

def parallel_eval_batched(sess, samples, input_name, label_name, input_shape, batch_size=None):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples = len(samples)
    n_jobs = cpu_count()

    if batch_size is None:
        batch_size = max(8, int(np.ceil(n_samples / n_jobs)))

    batches = [samples[i:i + batch_size] for i in range(0, n_samples, batch_size)]

    def evaluate_batch(batch):
        return black_box_batch(sess, batch, input_name, label_name, input_shape)

    results_nested = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(evaluate_batch)(batch) for batch in batches
    )

    return [val for sublist in results_nested for val in sublist]

def sobol_samples(dim, n_samples, cache_dir=".sobol_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sobol_d{dim}_n{n_samples}.npy")

    if os.path.isfile(cache_path):
        print(f"Loading cached Sobol samples from {cache_path}")
        unit_samples = np.load(cache_path)
    else:
        print(f"Generating new Sobol samples for d={dim}, n={n_samples}")
        sampler = qmc.Sobol(dim, scramble=False)
        unit_samples = sampler.random(n_samples)
        np.save(cache_path, unit_samples)

    return unit_samples

def optimizer(
    sess,
    input_name,
    label_name,
    input_shape,
    lower_bounds,
    upper_bounds,
    top_k=10,
    num_workers=cpu_count()
):
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)
    
    assert lower_bounds.shape == upper_bounds.shape, \
        "lower_bounds and upper_bounds must have the same shape"

    dim = len(lower_bounds)
    budget = min(2**15, max(4096, int(2**np.ceil(np.log2(100 * dim)))))
    if num_workers < 4:
        num_workers = 1

    print("Generating Sobol samples")
    unit_samples = sobol_samples(dim, budget)
    sobol_scaled = lower_bounds + unit_samples * (upper_bounds - lower_bounds)

    print("Evaluating samples using batched black-box")
    objective_values = parallel_eval_batched(
        sess, sobol_scaled, input_name, label_name, input_shape
    )
    
    sorted_indices = np.argsort(objective_values)
    topk_points = sobol_scaled[sorted_indices[:top_k]]

    best_lbfgs_val = float('inf')
    best_lbfgs_x = None
    total_lbfgs_time = 0.0

    print(f"Running L-BFGS-B from top-{top_k} Sobol samples")
    for i, init_point in enumerate(topk_points):
        print(f">> Init point {i+1}/{top_k}")
        start_lbfgs = time.time()

        def scalar_objective(x):
            x_reshaped = np.asarray(x, dtype=np.float32).reshape((1,) + tuple(input_shape))
            try:
                out = sess.run([label_name], {input_name: x_reshaped})[0]
            except TypeError:
                out = sess.run([label_name], {input_name: x_reshaped})[0]
            return float(out[0])

        res = minimize(
            scalar_objective,
            init_point,
            method='L-BFGS-B',
            bounds=list(zip(lower_bounds, upper_bounds)),
            options={'gtol': 1e-6, 'maxiter': 1000, 'eps': 1e-12}
        )
        lbfgs_time = time.time() - start_lbfgs
        total_lbfgs_time += lbfgs_time

        val_lbfgs = res.fun
        print(f"   L-BFGS-B result: {val_lbfgs:.6f} (Time: {lbfgs_time:.2f} s)")

        if val_lbfgs < best_lbfgs_val:
            best_lbfgs_val = val_lbfgs
            best_lbfgs_x = res.x

    return {
        "best_lbfgsb_val": best_lbfgs_val,
        "best_lbfgsb_x": best_lbfgs_x,
        "total_lbfgs_time": total_lbfgs_time,
    }
