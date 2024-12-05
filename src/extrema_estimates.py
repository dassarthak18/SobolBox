import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc

# We treat neural networks as a general MIMO black box
def black_box(sess, input_array, input_name, label_name):
	try:
		output_array = list(sess.run([label_name], {input_name: input_array.astype(np.float32)})[0][0])
	except TypeError:
		output_array = list(sess.run([label_name], {input_name: input_array.astype(np.float32)})[0])
	return output_array

# We use Latin Hypercube Sampling to generate a near-random sample for preliminary extremum estimation
def extremum_best_guess(sess, input_bounds):
	# get neural network metadata
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	# get the lower and upper input bounds
	lower_bounds = input_bounds[0]
	upper_bounds = input_bounds[1]
	# reshape if needed
	input_shape = sess.get_inputs()[0].shape
	for _ in range(len(input_shape)):
		if type(input_shape[_]) != int:
			input_shape[_] = 1
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
	'''for datapoint in sample_scaled:
		datapoint = np.reshape(datapoint, tuple(input_shape))
		sample_output.append(black_box(sess, datapoint, input_name, label_name))'''
	# compute the extrema estimates
	maxima = [max(x) for x in zip(*sample_output)]
	minima = [min(x) for x in zip(*sample_output)]
	return [minima, maxima]
