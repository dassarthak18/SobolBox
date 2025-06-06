import numpy as np
import multiprocessing as mp
from falsifier.extrema_estimates import black_box

def directional_slope(x, delta, sess, input_name, label_name, input_shape, lower_bounds, upper_bounds, num_directions=32):
    d = np.prod(input_shape)
    max_slope = 0.0
    fx = np.array(black_box(sess, x.reshape(input_shape), input_name, label_name, input_shape))
    
    for _ in range(num_directions):
        direction = np.random.normal(size=d)
        direction /= np.linalg.norm(direction)
        x_perturbed = np.clip(x + delta * direction, lower_bounds, upper_bounds)
        fx_perturbed = np.array(black_box(sess, x_perturbed.reshape(input_shape), input_name, label_name, input_shape))
        slope = np.linalg.norm(fx_perturbed - fx, ord=2) / delta
        if slope > max_slope:
            max_slope = slope

    return max_slope

def estimate_lipschitz_upper_bound(sess, input_shape, input_name, label_name, lower_bounds, upper_bounds, num_samples=2048, num_directions=32, delta=1e-2, safety_factor=1.2):
    d = np.prod(input_shape)
    lower_bounds = np.array(lower_bounds).flatten()
    upper_bounds = np.array(upper_bounds).flatten()

    def sample_input():
        return np.random.uniform(lower_bounds, upper_bounds)

    def worker(_):
        x = sample_input()
        return directional_slope(x, delta, sess, input_name, label_name, input_shape,
                                 lower_bounds, upper_bounds, num_directions=num_directions)

    with mp.Pool(mp.cpu_count()) as pool:
        all_slopes = pool.map(worker, range(num_samples))

    max_slope_sampled = max(all_slopes)

    # Apply safety factor to get a high-confidence upper bound
    lipschitz_upper_bound = max_slope_sampled * safety_factor

    return lipschitz_upper_bound
