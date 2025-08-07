import numpy as np
from joblib import Parallel, delayed
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

def optimize_extrema(sess, input_bounds, input_name, label_name, input_shape, i):
    # Minimize
    objective_min = create_objective_function(sess, input_shape, input_name, label_name, i, negate=False)
    result_min = optimize_1D(objective_min, input_bounds[0], input_bounds[1])

    # Maximize
    objective_max = create_objective_function(sess, input_shape, input_name, label_name, i, negate=True)
    result_max = optimize_1D(objective_max, input_bounds[0], input_bounds[1])

    return (
        result_min["best_lbfgsb_val"],
        result_max["best_lbfgsb_val"],  # Note: this is negative of actual max
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

    print("Running optimization per output dimension...\n")

    results = Parallel(n_jobs=min(cpu_count(), n_outputs), backend="loky")(
        delayed(optimize_extrema)(sess, input_bounds, input_name, label_name, input_shape, i)
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
