import numpy as np
import multiprocessing as mp
from falsifier.extrema_estimates import black_box

def slope_between_points(x1, x2, sess, input_name, label_name, input_shape):
    fx1 = np.array(black_box(sess, x1, input_name, label_name, input_shape))
    fx2 = np.array(black_box(sess, x2, input_name, label_name, input_shape))
    dist_input = np.linalg.norm(x2 - x1)
    if dist_input == 0:
        return 0.0
    dist_output = np.linalg.norm(fx2 - fx1)
    return dist_output / dist_input

def estimate_upper_lipschitz(sess, input_shape, input_name, label_name, lower_bounds, upper_bounds, num_samples=256, delta=1e-3, num_directions=16):
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    d = np.prod(input_shape)

    def sample_point():
        return np.random.uniform(lower_bounds, upper_bounds)

    def sample_pair():
        x = sample_point()
        # Random direction with norm scaled by delta (smaller steps for local slope)
        direction = np.random.randn(d)
        direction /= np.linalg.norm(direction)
        x_perturbed = x + delta * direction
        x_perturbed = np.clip(x_perturbed, lower_bounds, upper_bounds)
        return x, x_perturbed

    def worker(_):
        max_slope = 0.0
        for _ in range(num_directions):
            x1, x2 = sample_pair()
            slope = slope_between_points(x1, x2, sess, input_name, label_name, input_shape)
            if slope > max_slope:
                max_slope = slope
        return max_slope

    with mp.Pool(mp.cpu_count()) as pool:
        slopes = pool.map(worker, range(num_samples))

    return max(slopes)
