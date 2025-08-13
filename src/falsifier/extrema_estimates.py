import numpy as np
from joblib import Parallel, delayed, cpu_count
from falsifier.optimizer import optimize_1D

# Black box model runner
def black_box(sess, input_array, input_name, label_name, input_shape):
    flat_input = np.array(input_array, dtype=np.float32)
    reshaped_input = flat_input.reshape([
        dim if isinstance(dim, int) and dim > 0 else -1 for dim in input_shape
    ]).astype(np.float32)

    try:
        output = sess.run([label_name], {input_name: reshaped_input})[0][0]
    except TypeError:
        output = sess.run([label_name], {input_name: reshaped_input})[0]
    return output.tolist()

# Builds an objective function that extracts a specific output index
def create_objective_function(sess, input_shape, input_name, label_name, index, negate=False):
    def objective(x):
        val = black_box(sess, x, input_name, label_name, input_shape)[index]
        return -val if negate else val
    return objective

def determine_parallel_allocation(n_outputs, n_cores):
    if n_outputs > n_cores:
        x = n_outputs // n_cores
    else:
        x = n_cores
    y = n_cores // x

    n_jobs = min(x, y)
    n_threads = max(x, y)

    if n_jobs == 1 and n_cores // n_jobs >= 2:
        n_jobs *= 2
        n_threads //= 2

    while n_jobs * n_threads > n_cores and n_threads > 1:
        n_threads -= 1

    return n_jobs, n_threads

def optimize_extrema(sess, input_bounds, input_name, label_name, input_shape, i, inner_jobs):
    # Minimize
    objective_min = create_objective_function(sess, input_shape, input_name, label_name, i, negate=False)
    result_min = optimize_1D(objective_min, input_bounds[0], input_bounds[1], num_workers=inner_jobs)

    # Maximize
    objective_max = create_objective_function(sess, input_shape, input_name, label_name, i, negate=True)
    result_max = optimize_1D(objective_max, input_bounds[0], input_bounds[1], num_workers=inner_jobs)

    return (
        result_min["best_lbfgsb_val"],
        result_max["best_lbfgsb_val"],
        result_min["best_lbfgsb_x"],
        result_max["best_lbfgsb_x"],
    )

def extremum_refinement(sess, input_bounds):
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    lower_bounds = np.array(input_bounds[0])
    upper_bounds = np.array(input_bounds[1])
    input_bounds = (lower_bounds, upper_bounds)

    n_outputs = len(black_box(sess, lower_bounds, input_name, label_name, input_shape))
    total_cores = cpu_count()
    outer_jobs, inner_jobs = determine_parallel_allocation(n_outputs, total_cores)

    print(f"Using {outer_jobs} threads for outer parallelism")
    print(f"Using {inner_jobs} threads per inner optimization")

    # Capture inner_jobs for optimize_1D
    results = Parallel(n_jobs=outer_jobs, backend="threading")(
        delayed(optimize_extrema)(
            sess, input_bounds, input_name, label_name, input_shape, i, inner_jobs
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
        print(f"Output {i}: Min = {min_val:.6f}, Max = {-neg_max_val:.6f}")

    return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
