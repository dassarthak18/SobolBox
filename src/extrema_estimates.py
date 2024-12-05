import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import minimize, differential_evolution

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
	sample_output = []
	sample_output = Parallel(prefer="threads")(delayed(black_box)(sess, datapoint, input_name, label_name, input_shape) for datapoint in sample_scaled)
	#for datapoint in sample_scaled:
	#	sample_output.append(black_box(sess, datapoint, input_name, label_name, input_shape))
	# compute the extrema estimates
	minima = [min(x) for x in zip(*sample_output)]
	maxima = [max(x) for x in zip(*sample_output)]
	# compute inputs for those guesses
	minima_inputs = []
	for i in range(len(minima)):
		for j in range(len(sample_output)):
			if sample_output[j][i] == minima[i]:
				minima_inputs.append(sample_scaled[j])
				break
	maxima_inputs = []
	for i in range(len(maxima)):
		for j in range(len(sample_output)):
			if sample_output[j][i] == maxima[i]:
				maxima_inputs.append(sample_scaled[j])
				break
	return [minima_inputs, maxima_inputs, minima, maxima]

# Objective function generator for L-BFGS-B
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
		updated_minima_inputs = []
		updated_minima = []
		for index in range(len(minima_inputs)):
			objective = create_objective_function(sess, input_shape, input_name, label_name, index)
			result = minimize(objective, method = 'TNC', bounds=bounds, x0=list(minima_inputs[index]), options={'maxfun': 100, 'maxCGit': 10, 'eta': 0.2})
			if result.fun < minima[index]:
				updated_minima_inputs.append(list(result.x))
				updated_minima.append(result.fun)
			else:
				updated_minima_inputs.append(minima_inputs[index])
				updated_minima.append(minima[index])
		# refine the maxima estimate
		maxima_inputs = extremum_guess[1]
		maxima = extremum_guess[3]
		updated_maxima_inputs = []
		updated_maxima = []
		for index in range(len(maxima_inputs)):
			objective = create_objective_function(sess, input_shape, input_name, label_name, index, is_minima=False)
			result = minimize(objective, method = 'TNC', bounds=bounds, x0=list(maxima_inputs[index]), options={'maxfun': 100, 'maxCGit': 10, 'eta': 0.2})
			if result.fun > maxima[index]:
				updated_maxima_inputs.append(list(result.x))
				updated_maxima.append(result.fun)
			else:
				updated_maxima_inputs.append(maxima_inputs[index])
				updated_maxima.append(maxima[index])
		return [minima_inputs, maxima_inputs, minima, maxima]
	except ValueError:
		raise ValueError("Number of parameters too high, quitting gracefully.")
