import numpy as np
import onnxruntime
from falsifier.optimizer import optimize_1D

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

def create_objective_function(model_path, input_shape, input_name, label_name, index, negate=False):
    def objective(x):
        sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        val = black_box(sess, x, input_name, label_name, input_shape)[index]
        return -val if negate else val
    return objective

def extremum_refinement(model_path, input_bounds):
    dummy_sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = dummy_sess.get_inputs()[0].name
    label_name = dummy_sess.get_outputs()[0].name
    input_shape = dummy_sess.get_inputs()[0].shape

    lower_bounds = np.array(input_bounds[0])
    upper_bounds = np.array(input_bounds[1])

    n_outputs = len(black_box(dummy_sess, lower_bounds, input_name, label_name, input_shape))

    updated_minima = []
    updated_minima_inputs = []
    updated_maxima = []
    updated_maxima_inputs = []

    print("Running optimization per output dimension...\n")

    for i in range(n_outputs):
        print(f"=== Output {i} ===")

        # Minimize
        print("-> Minimizing")
        objective_min = create_objective_function(model_path, input_shape, input_name, label_name, i, negate=False)
        result_min = optimize_1D(objective_min, lower_bounds, upper_bounds)
        updated_minima.append(result_min["best_lbfgsb_val"])
        updated_minima_inputs.append(result_min["best_lbfgsb_x"])

        # Maximize
        print("-> Maximizing")
        objective_max = create_objective_function(model_path, input_shape, input_name, label_name, i, negate=True)
        result_max = optimize_1D(objective_max, lower_bounds, upper_bounds)
        updated_maxima.append(-result_max["best_lbfgsb_val"])
        updated_maxima_inputs.append(result_max["best_lbfgsb_x"])

        print(f"Min: {updated_minima[-1]:.6f}, Max: {updated_maxima[-1]:.6f}\n")

    return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
