import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import differential_evolution

# We treat neural networks as a general MIMO black box
def black_box(sess, input_array, input_name, label_name, input_shape):
	input_array = np.reshape(input_array, tuple(input_shape))
	try:
		output_array = list(sess.run([label_name], {input_name: input_array.astype(np.float32)})[0][0])
	except TypeError:
		output_array = list(sess.run([label_name], {input_name: input_array.astype(np.float32)})[0])
	return output_array

# We use Latin Hypercube Sampling to generate a near-random sample for preliminary extremum estimation
def extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape):
	# check no. of parameters, gracefully quit if necessary
	sampler = qmc.LatinHypercube(len(lower_bounds))
	num_parameters = np.product(input_shape)
	if num_parameters > 10**5:
		raise ValueError("Number of parameters too high, quitting gracefully.")
	else:
		d = int(10**5/num_parameters)
	# perform LHS to get a sample of input arrays within bounds
	sample = sampler.random(d)
	sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
	# compute the outputs
	sample_output = []
	sample_output = Parallel(prefer="threads")(delayed(black_box)(sess, datapoint, input_name, label_name, input_shape) for datapoint in sample_scaled)
	# compute the extrema estimates
	minima = [min(x) for x in zip(*sample_output)]
	maxima = [max(x) for x in zip(*sample_output)]
	return [minima, maxima]

# Objective function generator for differential evolution
def create_objective_function(sess, input_shape, input_name, label_name, index, target_value, is_minima=True):
	def objective(x):
		arr = black_box(sess, x, input_name, label_name, input_shape)
		if is_minima:
			return -1 * (target_value - arr[index])
		else:
			return target_value - arr[index]
	return objective

# We use Differential Evolution to refine our LHS extremum estimates
def diff_evo_estimates(sess, input_bounds):
	# get neural network metadata
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	# reshape if needed
	input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
	# get the lower and upper input bounds
	lower_bounds = input_bounds[0]
	upper_bounds = input_bounds[1]
	# get the preliminary estimates
	try:
		extemum_guess = extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape)
	except ValueError:
		raise ValueError("Number of parameters too high, quitting gracefully.")
	bounds = list(zip(lower_bounds, upper_bounds))
	# refine the minima estimate
	minima = extremum_guess[0]
	updated_minima = []
	for index in range(len(minima)):
		objective = create_objective_function(sess, input_shape, input_name, label_name, index, minima[index])
  		result = differential_evolution(objective, bounds=bounds)
  		updated_minima.append(minima[index]+result.fun)
	# refine the maxima estimate
	maxima = extremum_guess[1]
	updated_maxima = []
	for index in range(len(maxima)):
		objective = create_objective_function(sess, input_shape, input_name, label_name, index, maxima[index], is_minima=False)
		result = differential_evolution(objective, bounds=bounds)
  		updated_maxima.append(maxima[index]-result.fun)
	return [updated_minima, updated_maxima]
