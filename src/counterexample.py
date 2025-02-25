import numpy as np
import copy, csv, ast
from extrema_estimates import black_box
from z3 import *

def validateCE(model, sess):
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  # reshape if needed
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
  
  x_decls = sorted([str(d) for d in model.decls() if "X_" in d.name()])
  y_decls = sorted([str(d) for d in model.decls() if "Y_" in d.name()])
  input_array = [float(model.eval(Real(d)).as_decimal(20)) for d in x_decls]
  
  output_array_pred = [float(model.eval(Real(d)).as_decimal(20)) for d in y_decls]
  output_array_true = black_box(sess, input_array, input_name, label_name, input_shape)

  if np.allclose(output_array_pred, output_array_true, rtol=0, atol=1e-15):
    return True
  return False

def SAT_check(solver, sess, filename, input_lb, input_ub):
  print("Checking for violations in LHS samples.")
        
  if str(solver.check()) == "unsat":
    print("No safety violations found.")
    return "holds"
    
  model = solver.model()
  variables = sorted([str(d) for d in model.decls()])
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  # reshape if needed
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]

  LHSCacheFile = "../cache/" + filename[:-5] + "_lhs.csv"
  with open(LHSCacheFile, mode='r', newline='') as cacheFile:
    reader = csv.reader(cacheFile, delimiter='|')
    for row in reader:
      fetched_input_lb = ast.literal_eval(row[0])
      fetched_input_ub = ast.literal_eval(row[1])
      if input_lb == fetched_input_lb and input_ub == fetched_input_ub:
        input_array = ast.literal_eval(row[2])
        output_array = ast.literal_eval(row[3])
        break

  for i in range(len(input_array)):
    vals_lb = np.concatenate((input_array[i], output_array[i]))
    solver_2 = copy.deepcopy(solver)
    for j in range(len(variables)):
      solver_2.add(Real(variables[j]) == vals_lb[j])
    if str(solver_2.check()) == "sat":
      model = solver_2.model()
      s = "violated\nCE: "
      for i in range(len(variables)):
        val = float(model.eval(Real(variables[i])).as_decimal(20))
        s += variables[i] + " = " + str(val) + "\n"
      print("Safety violation detected.")
      return s

  print("Inconclusive analysis.")
  return "unknown"
