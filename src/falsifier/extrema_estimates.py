import numpy as np
import onnxruntime
from falsifier.optimizer import optimize_1D

def batched_black_box(sess, batch_input, input_name, label_name):
    batch_input = np.array(batch_input, dtype=np.float32)
    return sess.run([label_name], {input_name: batch_input})[0]

def create_objective_function(sess, input_shape, input_name, label_name, index, negate=False):
    def objective(x):
        reshaped = np.array(x, dtype=np.float32).reshape(input_shape)
        val = sess.run([label_name], {input_name: reshaped})[0][0][index]
        return -val if negate else val
    return objective

def extremum_refinement(model_path, input_bounds):
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    lower_bounds = np.array(input_bounds[0])
    upper_bounds = np.array(input_bounds[1])

    dummy_input = np.array(lower_bounds, dtype=np.float32).reshape(input_shape)
    dummy_output = sess.run([label_name], {input_name: dummy_input})[0][0]
    n_outputs = len(dummy_output)

    updated_minima = []
    updated_maxima = []
    updated_minima_inputs = []
    updated_maxima_inputs = []

    for i in range(n_outputs):
        print(f"=== Output {i} ===")
        print("-> Minimizing")
        objective_min = create_objective_function(sess, input_shape, input_name, label_name, i, negate=False)
        result_min = optimize_1D(objective_min, lower_bounds, upper_bounds)
        updated_minima.append(result_min["best_lbfgsb_val"])
        updated_minima_inputs.append(result_min["best_lbfgsb_x"])

        print("-> Maximizing")
        objective_max = create_objective_function(sess, input_shape, input_name, label_name, i, negate=True)
        result_max = optimize_1D(objective_max, lower_bounds, upper_bounds)
        updated_maxima.append(-result_max["best_lbfgsb_val"])
        updated_maxima_inputs.append(result_max["best_lbfgsb_x"])

        print(f"Min: {updated_minima[-1]:.6f}, Max: {updated_maxima[-1]:.6f}\n")

    return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
