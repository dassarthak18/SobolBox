from memo_store import memo

import hashlib
import numpy as np
from joblib import Parallel, delayed, cpu_count
from falsifier.optimizer import sobol_samples, optimize_1D

# Black box model runner
def black_box(sess, input_array, input_name, label_name, input_shape, onnxFile):
    hash = hashlib.md5(input_array.tobytes()).hexdigest()
    if onnxFile in memo:
        if hash in memo[onnxFile]:
            return memo[onnxFile][hash]
    flat_input = np.array(input_array, dtype=np.float32)
    reshaped_input = flat_input.reshape([
        dim if isinstance(dim, int) and dim > 0 else -1 for dim in input_shape
    ]).astype(np.float32)

    try:
        output = sess.run([label_name], {input_name: reshaped_input})[0][0]
    except TypeError:
        output = sess.run([label_name], {input_name: reshaped_input})[0]
    result = output.tolist()
    if onnxFile in memo:
        memo[onnxFile][hash] = result
    else:
        memo[onnxFile] = {}
        memo[onnxFile][hash] = result
    return result

# Builds an objective function that extracts a specific output index
def create_objective_function(sess, input_shape, input_name, label_name, index, onnxFile, negate=False):
    def objective(x):
        val = black_box(sess, x, input_name, label_name, input_shape, onnxFile)[index]
        return -val if negate else val
    return objective

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

def optimize_extrema(sess, input_bounds, input_name, label_name, input_shape, i, objective_mins, objective_maxs, topk_mins, topk_maxs, onnxFile):
    # Minimize
    result_min = optimize_1D(objective_mins[i], input_bounds[0], input_bounds[1], topk_mins[i])

    # Maximize
    result_max = optimize_1D(objective_maxs[i], input_bounds[0], input_bounds[1], topk_maxs[i])

    return (
        result_min["best_lbfgsb_val"],
        result_max["best_lbfgsb_val"],
        result_min["best_lbfgsb_x"],
        result_max["best_lbfgsb_x"],
    )

def extremum_refinement(sess, input_bounds, onnxFile):
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    lower_bounds = np.array(input_bounds[0])
    upper_bounds = np.array(input_bounds[1])
    input_bounds = (lower_bounds, upper_bounds)

    dim = len(lower_bounds)
    budget = min(2**20, max(8192, int(2**np.ceil(np.log2(500 * dim)))))
    top_k = max(100, int(np.ceil(0.1 * budget)))
    center_point = 0.5 * (lower_bounds + upper_bounds)
    unit_samples = sobol_samples(dim, budget)
    sobol_scaled = lower_bounds + unit_samples * (upper_bounds - lower_bounds)
    n_outputs = len(black_box(sess, lower_bounds, input_name, label_name, input_shape, onnxFile))

    objective_mins = []
    topk_mins = []
    
    for i in range(n_outputs):
        objective_mins.append(create_objective_function(sess, input_shape, input_name, label_name, i, onnxFile))
        objective_values = np.array(parallel_eval(objective_mins[-1], sobol_scaled))
        sorted_indices = np.argsort(objective_values)
        topk_points = sobol_scaled[sorted_indices[:top_k]]
        if objective_mins[-1](center_point) < objective_values[sorted_indices[top_k - 1]]:
            topk_points = np.vstack([topk_points, center_point])
        topk_mins.append(topk_points)

    objective_maxs = []
    topk_maxs = []
    
    for i in range(n_outputs):
        objective_maxs.append(create_objective_function(sess, input_shape, input_name, label_name, i, onnxFile, negate=True))
        objective_values = np.array(parallel_eval(objective_maxs[-1], sobol_scaled))
        sorted_indices = np.argsort(objective_values)
        topk_points = sobol_scaled[sorted_indices[:top_k]]
        if objective_maxs[-1](center_point) < objective_values[sorted_indices[top_k - 1]]:
            topk_points = np.vstack([topk_points, center_point])
        topk_maxs.append(topk_points)
        
    results = Parallel(n_jobs=cpu_count(), backend="threading")(
        delayed(optimize_extrema)(
            sess, input_bounds, input_name, label_name, input_shape, i, objective_mins, objective_maxs, topk_mins, topk_maxs, onnxFile
        )
        for i in range(n_outputs)
    )

    updated_minima = []
    updated_maxima = []
    updated_minima_inputs = []
    updated_maxima_inputs = []

    for i, (min_val, neg_max_val, min_x, max_x) in enumerate(results):
        updated_minima.append(min_val)
        updated_maxima.append(-neg_max_val)
        updated_minima_inputs.append(min_x)
        updated_maxima_inputs.append(max_x)
        print(f"Output {i}: Min = {min_val}, Max = {-neg_max_val}")

    return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
