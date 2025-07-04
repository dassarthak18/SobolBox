import numpy as np
import copy, csv, ast, json
#import concurrent.futures
import pymc as pm
import pytensor.tensor as pt
from falsifier.extrema_estimates import black_box
from z3 import *

def all_in_elementwise_range(arr, lowers, uppers):
  arr = np.asarray(arr)
  lowers = np.asarray(lowers)
  uppers = np.asarray(uppers)
  return np.all((arr >= lowers) & (arr <= uppers))

def validateCE(model, sess, input_array, input_lb, input_ub):
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  input_shape = sess.get_inputs()[0].shape
  if not all_in_elementwise_range(input_array, input_lb, input_ub):
    print("Candidate CE invalidated - invalid input range.")
    return False
  y_decls = sorted([str(d) for d in model.decls() if "Y_" in d.name()])
  output_array_pred = [float(model.eval(Real(d)).as_decimal(100)) for d in y_decls]
  output_array_true = black_box(sess, input_array, input_name, label_name, input_shape)
  if not np.allclose(output_array_pred, output_array_true, rtol=0, atol=1e-15):
    print("Candidate CE invalidated - invalid outputs.")
    return False
  print("Candidate CE validated.")
  return True

def CE_sampler_advi(sess, lower, upper, targets, input_shape, sigma=0.1):
    inputsize = input_shape[-1]
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    targets = np.array(targets)
    lower = np.array(lower)
    upper = np.array(upper)
    inputsize = len(lower)
    sigma2 = sigma ** 2
    with pm.Model() as model:
        z = pm.Normal("z", mu=0, sigma=1, shape=inputsize)
        x = pm.Deterministic("x", lower + (upper - lower) * pm.math.sigmoid(z))
        def logp_fn(x_val):
            x_exp = pt.reshape(x_val, (1, -1))
            diffs = x_exp - targets
            sq_dists = pm.math.sum(pm.math.sqr(diffs), axis=1)
            logps = -0.5 * sq_dists / sigma2
            return pm.math.logsumexp(logps)
        pm.Potential("target_bias", logp_fn(x))
        approx = pm.fit(n=10000, method="advi")
        n_samples = 10*np.min([int(2**20), np.max([4096, int(2**np.floor(np.log2(500*inputsize)))])])
        posterior_samples = approx.sample(n_samples, random_seed=42)
    samples = posterior_samples.posterior["x"].stack(sample=("chain", "draw")).values.T
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    outputs = list(executor.map(lambda sample: black_box(sess, sample, input_name, label_name, input_shape), samples))
    outputs = [black_box(sess, sample, input_name, label_name, input_shape) for sample in samples]
    dists = [np.min(np.linalg.norm(sample - targets, axis=1)) for sample in samples]
    sorted_indices = np.argsort(dists)
    samples = samples[sorted_indices]
    outputs = [outputs[i] for i in sorted_indices]
    return samples, outputs

def unknown_CE_check(sess, solver_2, input_lb, input_ub, optimas, input_shape):
  print("Computing ADVI samples.")
  X, Y = CE_sampler_advi(sess, input_lb, input_ub, optimas, input_shape)
  print("Checking for violations in ADVI samples.")
  X_vars = [Real(f"X_{i}") for i in range(len(X[0]))]
  Y_vars = [Real(f"Y_{i}") for i in range(len(Y[0]))]
  for i in range(len(Y)):
    solver_2.push()
    for j in range(len(X_vars)):
      solver_2.add(X_vars[j] == X[i][j])
    for j in range(len(Y_vars)):
      solver_2.add(Y_vars[j] == Y[i][j])
    if str(solver_2.check()) == "sat":
      #model = solver_2.model()
      #if not validateCE(model, sess, X[i], input_lb, input_ub):
      #  continue
      s = "sat"
      for k in range(len(X[i])):
        if k == 0:
          s += "\n(("
        else:
          s += "\n ("
        s += str(X_vars[k]) + " " + str(X[i][k]) + ")"
      for k in range(len(Y[i])):
        s += "\n (" + str(Y_vars[k]) + " " + str(Y[i][k]) + ")"
      s += ")"
      print("Safety violation detected in ADVI samples.")
      return s
    solver_2.pop()
  print("No safety violations found.")
  return "unknown"

def nearest_optima_distance(sample, optima_array):
  return min(np.linalg.norm(sample - np.array(opt)) for opt in optima_array)

def SAT_check(solver, solver_2, sess, input_lb, input_ub, output_lb_inputs, output_ub_inputs, setting):
  if str(solver.check()) == "unsat":
    print("No safety violations found.")
    return "unsat"
  print("Checking for violations in optima.")
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  input_shape = sess.get_inputs()[0].shape
  input_array = []
  for i in range(len(output_lb_inputs)):
    input_array.append(output_lb_inputs[i])
    input_array.append(output_ub_inputs[i])
  #with concurrent.futures.ThreadPoolExecutor() as executor:
  #  output_array = list(executor.map(lambda x: black_box(sess, x, input_name, label_name, input_shape), input_array))
  output_array = [black_box(sess, input, input_name, label_name, input_shape) for input in input_array]
  X_vars = [Real(f"X_{i}") for i in range(len(input_array[0]))]
  Y_vars = [Real(f"Y_{i}") for i in range(len(output_array[0]))]
  variables = []
  for i in range(len(output_array[0])):
    variables.append(f"Y_{i}")
  for i in range(len(input_array)):
    solver_2.push()
    for j in range(len(X_vars)):
      solver_2.add(X_vars[j] == input_array[i][j])
    for j in range(len(Y_vars)):
      solver_2.add(Y_vars[j] == output_array[i][j])
    if str(solver_2.check()) == "sat":
      #model = solver_2.model()
      #if not validateCE(model, sess, input_array[i], input_lb, input_ub):
      #  continue
      s = "sat"
      for k in range(len(input_array[i])):
        if k == 0:
          s += "\n(("
        else:
          s += "\n ("
        s += str(X_vars[k]) + " " + str(input_array[i][k]) + ")"
      for k in range(len(output_array[i])):
        s += "\n (" + str(Y_vars[k]) + " " + str(output_array[i][k]) + ")"
      s += ")"
      print("Safety violation detected in optima.")
      return s
    solver_2.pop()
  print("Checking for violations in Sobol sequence samples.")
  LHSCacheFile = "../cache/sobol.csv"
  with open(LHSCacheFile, mode='r', newline='') as cacheFile:
    reader = csv.reader(cacheFile, delimiter='|')
    for row in reader:
      if row[0] == str(len(input_lb)):
        sample = json.loads(row[1])
        break
  input_lb = np.array(input_lb)
  input_ub = np.array(input_ub)
  input_array = input_lb + sample * (input_ub - input_lb)
  optima_array = output_lb_inputs + output_ub_inputs
  sample_dist_pairs = [(inp, nearest_optima_distance(inp, optima_array)) for inp in input_array]
  sample_dist_pairs.sort(key=lambda x: x[1])
  input_array = [pair[0] for pair in sample_dist_pairs]
  #with concurrent.futures.ThreadPoolExecutor() as executor:
  #  output_array = list(executor.map(lambda x: black_box(sess, x, input_name, label_name, input_shape), input_array))
  output_array = [black_box(sess, input, input_name, label_name, input_shape) for input in input_array]
  for i in range(len(input_array)):
    solver_2.push()
    for j in range(len(X_vars)):
      solver_2.add(X_vars[j] == input_array[i][j])
    for j in range(len(Y_vars)):
      solver_2.add(Y_vars[j] == output_array[i][j])
    if str(solver_2.check()) == "sat":
      #model = solver_2.model()
      #if not validateCE(model, sess, input_array[i], input_lb, input_ub):
      #  continue
      s = "sat"
      for k in range(len(input_array[i])):
        if k == 0:
          s += "\n(("
        else:
          s += "\n ("
        s += str(X_vars[k]) + " " + str(input_array[i][k]) + ")"
      for k in range(len(output_array[i])):
        s += "\n (" + str(Y_vars[k]) + " " + str(output_array[i][k]) + ")"
      s += ")"
      print("Safety violation detected in Sobol sequence.")
      return s
    solver_2.pop()
  if setting:
    second_pass = unknown_CE_check(sess, solver_2, input_lb, input_ub, output_lb_inputs+output_ub_inputs, input_shape)
    return second_pass
  print("Inconclusive analysis.")
  return "unknown"
