import numpy as np
import csv, ast, json, warnings
from pathlib import Path
#from pyDOE3 import lhs
from scipy.stats import qmc
from scipy.optimize import minimize, SR1

# We treat neural networks as a general MIMO black box
def black_box(sess, input_array, input_name, label_name, input_shape):
    fixed_shape = []
    flat_input = np.array(input_array, dtype=np.float32)
    for dim in input_shape:
        fixed_shape.append(dim if isinstance(dim, int) and dim > 0 else -1)
    reshaped_input = flat_input.reshape(fixed_shape).astype(np.float32)
    try:
        value = sess.run([label_name], {input_name: reshaped_input})[0][0]
    except TypeError:
        value = sess.run([label_name], {input_name: reshaped_input})[0]
    return value.tolist()

# We use Sobol sequence sampling to generate a near-random sample for preliminary extremum estimation
def extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape):
	print("Computing Sobol sequence samples.")
	# check no. of parameters, gracefully quit if necessary
	inputsize = len(lower_bounds)
	#n_samples = 20*inputsize
	n_samples = np.min([int(2**20), np.max([4096, int(2**np.floor(np.log2(500*inputsize)))])])
	print(f"Calculating Sobol sequence for {n_samples} samples.")
	lower_bounds = np.array(lower_bounds)
	upper_bounds = np.array(upper_bounds)

	LHSCacheFile = "../cache/sobol.csv"
	cacheFound = False
	if Path(LHSCacheFile).exists():
		with open(LHSCacheFile, mode='r', newline='') as cacheFile:
			reader = csv.reader(cacheFile, delimiter='|')
			for row in reader:
				if row[0] == str(inputsize):
					#sample = ast.literal_eval(row[1])
					sample = json.loads(row[1])
					cacheFound = True
					print("Retrieved Sobol sequence from cache.")
					break

	if not cacheFound:
		#sample = lhs(inputsize, samples=n_samples, criterion='lhsmu')
		#sample_scaled = lower_bounds + sample * (upper_bounds - lower_bounds)
		sampler = qmc.Sobol(inputsize, scramble=False)
		sample = sampler.random(n_samples)
		
	try:
		sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
	except ValueError:
		sample_scaled = lower_bounds + sample * (upper_bounds - lower_bounds)
	
	# compute the outputs
	sample_output = []
	for datapoint in sample_scaled:
		sample_output.append(black_box(sess, datapoint, input_name, label_name, input_shape))

	# cache the Sobol sequence inputs and outputs for future use
	if not cacheFound:
		with open(LHSCacheFile, mode='a', newline='') as cacheFile:
			writer = csv.writer(cacheFile, delimiter='|')
			if not Path(LHSCacheFile).exists():
	        		writer.writerow(["input_size", "unscaled_sample"])
			writer.writerow([str(inputsize), json.dumps(sample.tolist())])
	
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

# We use L-BFGS-B to refine our Sobol sequence extremum estimates
def extremum_refinement(sess, input_bounds, filename):
	# get neural network metadata
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	# reshape if needed
	input_shape = sess.get_inputs()[0].shape
	# get the lower and upper input bounds
	lower_bounds = input_bounds[0]
	upper_bounds = input_bounds[1]
	# get the preliminary estimates
	extremum_guess = extremum_best_guess(sess, lower_bounds, upper_bounds, input_name, label_name, input_shape)
	bounds = list(zip(lower_bounds, upper_bounds))
	print("Refining the Sobol sequence samples.")
	# refine the minima estimate
	minima_inputs = extremum_guess[0]
	minima = extremum_guess[2]
	updated_minima = []
	updated_minima_inputs = []
	for index in range(len(minima)):
		objective = create_objective_function(sess, input_shape, input_name, label_name, index)
		x0 = list(minima_inputs[index])
		result = minimize(objective, method = 'L-BFGS-B', bounds = bounds, x0 = x0, options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'eps': 1e-12})
		'''try:
			with warnings.catch_warnings(record=True) as w:
				warnings.simplefilter("always", UserWarning)
				result = minimize(objective, method = 'trust-constr', bounds = bounds, x0 = x0, jac = '2-point', hess = SR1(), options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'xtol': 1e-12})
				for warning in w:
					if "delta_grad == 0.0" in str(warning.message):
						raise RuntimeError("Switch to zero Hessian")
		except RuntimeError:
			result = minimize(objective, method = 'trust-constr', bounds = bounds, x0 = x0, jac = '2-point', hess=lambda x: np.zeros((len(x), len(x))), options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'xtol': 1e-12})'''
		if result.fun > minima[index]:
			updated_minima.append(minima[index])
			updated_minima_inputs.append(x0)
		else:
			updated_minima.append(result.fun)
			updated_minima_inputs.append(list(result.x))
	# refine the maxima estimate
	maxima_inputs = extremum_guess[1]
	maxima = extremum_guess[3]
	updated_maxima = []
	updated_maxima_inputs = []
	for index in range(len(maxima)):
		objective = create_objective_function(sess, input_shape, input_name, label_name, index, is_minima=False)
		x0 = list(maxima_inputs[index])
		result = minimize(objective, method = 'L-BFGS-B', bounds = bounds, x0 = x0, options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'eps': 1e-12})
		'''try:
			with warnings.catch_warnings(record=True) as w:
				warnings.simplefilter("always", UserWarning)
				result = minimize(objective, method = 'trust-constr', bounds = bounds, x0 = x0, jac = '2-point', hess = SR1(), options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'xtol': 1e-12})
				for warning in w:
					if "delta_grad == 0.0" in str(warning.message):
						raise RuntimeError("Switch to zero Hessian")
		except RuntimeError:
			result = minimize(objective, method = 'trust-constr', bounds = bounds, x0 = x0, jac = '2-point', hess=lambda x: np.zeros((len(x), len(x))), options = {'disp': False, 'gtol': 1e-6, 'maxiter': 300, 'xtol': 1e-12})'''
		if -result.fun < maxima[index]:
			updated_maxima.append(maxima[index])
			updated_maxima_inputs.append(x0)
		else:
			updated_maxima.append(-result.fun)
			updated_maxima_inputs.append(list(result.x))
	print("Output bounds extracted.")
	# cache the computer bounds for future use
	boundsCacheFile = "../cache/" + filename[:-5] + "_bounds.csv"
	with open(boundsCacheFile, mode='a', newline='') as cacheFile:
		writer = csv.writer(cacheFile, delimiter='|')
		if not Path(boundsCacheFile).exists():
			writer.writerow(["input_lb", "input_ub", "output_lb", "output_ub", "minima_inputs", "maxima_inputs"])
		writer.writerow([json.dumps(lower_bounds), json.dumps(upper_bounds), json.dumps(updated_minima), json.dumps(updated_maxima), json.dumps(updated_minima_inputs), json.dumps(updated_maxima_inputs)])
		
	return [updated_minima, updated_maxima, updated_minima_inputs, updated_maxima_inputs]
