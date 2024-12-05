import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import minimize

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
	num_parameters = np.prod(input_shape)
	if num_parameters > 10**5:
		raise ValueError("Number of parameters too high, quitting gracefully.")
	else:
		d = int(10**6/num_parameters)
	# perform LHS to get a sample of input arrays within bounds
	sample = sampler.random(d)
	sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
	# compute the outputs
	sample_output = Parallel(n_jobs=-1, prefer="threads", require="sharedmem")(
		delayed(black_box)(sess, datapoint, input_name, label_name, input_shape) for datapoint in sample_scaled
	)
	minima = [min(x) for x in zip(*sample_output)]
	maxima = [max(x) for x in zip(*sample_output)]
	# compute inputs for those guesses
	minima_inputs = [sample_scaled[j] for i in range(len(minima)) for j in range(len(sample_output)) if sample_output[j][i] == minima[i]]
	maxima_inputs = [sample_scaled[j] for i in range(len(maxima)) for j in range(len(sample_output)) if sample_output[j][i] == maxima[i]]
	return [minima_inputs, maxima_inputs, minima, maxima]

# Objective function generator for TNC
def create_objective_function(sess, input_shape, input_name, label_name, index, is_minima=True):
	def objective(x):
		arr = black_box(sess, x, input_name, label_name, input_shape)
		if is_minima:
			return arr[index]
		else:
			return -1*arr[index]
	return objective

# We use TNC to refine our LHS extremum estimates
def extremum_refinement(sess, input_bounds):
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
		extremum_guess = extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape)
		bounds = list(zip(lower_bounds, upper_bounds))
		# refine the minima estimate
		minima_inputs = extremum_guess[0]
		minima = extremum_guess[2]
		results_minima = Parallel(n_jobs=-1, prefer="threads", require="sharedmem")(
			delayed(minimize)(
				create_objective_function(sess, input_shape, input_name, label_name, index),
				method='TNC',
				bounds=bounds,
				x0=list(minima_inputs[index]),
			) for index in range(len(minima_inputs))
		)
		updated_minima_inputs = [list(result.x) for result in results_minima]
		updated_minima = [result.fun for result in results_minima]
		# refine the maxima estimate
		maxima_inputs = extremum_guess[1]
		maxima = extremum_guess[3]
		results_maxima = Parallel(n_jobs=-1, prefer="threads", require="sharedmem")(
			delayed(minimize)(
				create_objective_function(sess, input_shape, input_name, label_name, index, is_minima=False),
				method='TNC',
				bounds=bounds,
				x0=list(maxima_inputs[index]),
			) for index in range(len(maxima_inputs))
		)
		updated_maxima_inputs = [list(result.x) for result in results_maxima]
		updated_maxima = [-result.fun for result in results_maxima]
		return [minima_inputs, maxima_inputs, minima, maxima]
	except ValueError:
		raise ValueError("Number of parameters too high, quitting gracefully.")
