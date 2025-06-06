import numpy as np
import multiprocessing as mp
from falsifier.extrema_estimates import black_box

def directional_slope(x, delta, sess, input_name, label_name, input_shape, num_directions=8):
    d = len(x)
    max_slope = 0.0
    fx = np.array(black_box(sess, x, input_name, label_name, input_shape))
    
    for _ in range(num_directions):
        direction = np.random.choice([-1.0, 1.0], size=d)
        direction /= np.linalg.norm(direction)
        x_perturbed = np.clip(x + delta * direction, 0.0, 1.0)  # Modify bounds if needed
        fx_perturbed = np.array(black_box(sess, x_perturbed, input_name, label_name, input_shape))
        slope = np.linalg.norm(fx_perturbed - fx, ord=2) / delta
        max_slope = max(max_slope, slope)

    return max_slope

def estimate_lower_lipschitz(sess, input_shape, input_name, label_name, lower_bounds, upper_bounds, num_samples=256, num_directions=16, delta=1e-2):
    d = np.prod(input_shape)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    def sample_input():
        return np.random.uniform(lower_bounds, upper_bounds)

    def worker(_):
        x = sample_input()
        return directional_slope(x, delta, sess, input_name, label_name, input_shape, num_directions=num_directions)

    with mp.Pool(mp.cpu_count()) as pool:
        all_slopes = pool.map(worker, range(num_samples))

    return max(all_slopes)
