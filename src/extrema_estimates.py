import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc
from scipy.optimize import differential_evolution

# We treat neural networks as a general MIMO black box
def black_box(sess, input_array, input_name, label_name):
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
		return "unknown"
	else:
		d = int(10**5/num_parameters)
	# perform LHS to get a sample of input arrays within bounds
	sample = sampler.random(d)
	sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
	# compute the outputs
	sample_output = []
	sample_output = Parallel(prefer="threads")(delayed(black_box)(sess, np.reshape(datapoint, tuple(input_shape)), input_name, label_name) for datapoint in sample_scaled)
	# compute the extrema estimates
	minima = [min(x) for x in zip(*sample_output)]
	maxima = [max(x) for x in zip(*sample_output)]
	return [minima, maxima]

# We use Differential Evolution to refine our LHS extremum estimates
def diff_evo_estimates(sess, input_bounds):
	# get neural network metadata
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	# reshape if needed
	input_shape = sess.get_inputs()[0].shape
	for _ in range(len(input_shape)):
		if type(input_shape[_]) != int:
			input_shape[_] = 1
	# get the lower and upper input bounds
	lower_bounds = input_bounds[0]
	upper_bounds = input_bounds[1]
	# get the preliminary estimates
	extemum_guess = extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape)
	if extremum_guess == "unknown":
		return "unknown"
	bounds = list(zip(lower_bounds, upper_bounds))
	# refine the minima estimate
	minima = extremum_guess[0]
	updated_minima = []
	for index in range(len(minima)):
  		def objective(x):
      			arr = black_box(sess, np.reshape(x, tuple(input_shape)), input_name, label_name)
      			return -1*(minima[index]-arr[index])
		result = differential_evolution(objective, bounds=bounds)
  		updated_minima.append(minima[index]+result.fun)
	# refine the maxima estimate
	maxima = extremum_guess[1]
	updated_maxima = []
	for index in range(len(maxima)):
  		def objective(x):
      			arr = black_box(sess, np.reshape(x, tuple(input_shape)), input_name, label_name)
      			return (maxima[index]-arr[index])
		result = differential_evolution(objective, bounds=bounds)
  		updated_maxima.append(maxima[index]-result.fun)
	return [updated_minima, updated_maxima]
