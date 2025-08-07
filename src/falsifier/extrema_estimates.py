import numpy as np
import torch
from falsifier.optimizer import optimize_1D

def run_pytorch(model, input_tensor):
    with torch.no_grad():
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        output = model(input_tensor)
        return output.numpy()

def create_objective_function(model, input_shape, output_index, negate=False):
    def objective(x):
        reshaped = np.array(x, dtype=np.float32).reshape(input_shape)
        output = run_pytorch(model, reshaped)[0][output_index]
        return -output if negate else output
    return objective

def extremum_refinement(model, input_bounds):
    model.eval()

    input_shape = (1, len(input_bounds[0]))  # Assumes flat input
    lower_bounds = np.array(input_bounds[0])
    upper_bounds = np.array(input_bounds[1])

    dummy_input = np.array(lower_bounds, dtype=np.float32).reshape(input_shape)
    dummy_output = run_pytorch(model, dummy_input)[0]
    n_outputs = len(dummy_output)

    updated_minima = []
    updated_maxima = []
    updated_minima_inputs = []
    updated_maxima_inputs = []

    for i in range(n_outputs):
        print(f"=== Output {i} ===")
        print("-> Minimizing")
        objective_min = create_objective_function(model, input_shape, i, negate=False)
        result_min = optimize_1D(objective_min, lower_bounds, upper_bounds)
        updated_minima.append(result_min["best_lbfgsb_val"])
        updated_minima_inputs.append(result_min["best_lbfgsb_x"])

        print("-> Maximizing")
        objective_max = create_objective_function(model, input_shape, i, negate=True)
        result_max = optimize_1D(objective_max, lower_bounds, upper_bounds)
        updated_maxima.append(-result_max["best_lbfgsb_val"])
        updated_maxima_inputs.append(result_max["best_lbfgsb_x"])

        print(f"Min: {updated_minima[-1]:.6f}, Max: {updated_maxima[-1]:.6f}\n")

    return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
