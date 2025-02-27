import numpy as np
import copy, csv, ast, json
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

def SAT_check(solver, solver_2, sess, input_lb, input_ub, output_lb_inputs, output_ub_inputs):
  print("Checking for violations in Sobol sequence samples.")
        
  if str(solver.check()) == "unsat":
    print("No safety violations found.")
    return "holds"
    
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  # reshape if needed
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]

  input_array = []
  for i in range(len(output_lb_inputs)):
    input_array.append(output_lb_inputs[i])
    input_array.append(output_ub_inputs[i])
  output_array = []
  for datapoint in input_array:
    output_array.append(black_box(sess, datapoint, input_name, label_name, input_shape))

  variables = []
  for i in range(len(output_array[0])):
    variables.append(f"Y_{i}")

  for i in range(len(input_array)):
    solver_2.push()
    for j in range(len(variables)):
      solver_2.add(Real(variables[j]) == output_array[i][j])
    if str(solver_2.check()) == "sat":
      model = solver_2.model()
      s = "violated\nCE: "
      for k in range(len(input_array[i])):
        s += "X_" + str(k) + " = " + str(input_array[i][k]) + "\n"
      for k in range(len(variables)):
        val = float(model.eval(Real(variables[k])).as_decimal(32))
        s += variables[k] + " = " + str(val) + "\n"
      print("Safety violation detected in optima.")
      return s
    solver_2.pop()
  
  LHSCacheFile = "../cache/sobol.csv"
  with open(LHSCacheFile, mode='r', newline='') as cacheFile:
    reader = csv.reader(cacheFile, delimiter='|')
    for row in reader:
      if row[0] == str(len(input_lb)):
        #sample = ast.literal_eval(row[1])
        sample = json.loads(row[1])
        break
  input_lb = np.array(input_lb)
  input_ub = np.array(input_ub)
  input_array = input_lb + sample * (input_ub - input_lb)
  output_array = []
  for datapoint in input_array:
    output_array.append(black_box(sess, datapoint, input_name, label_name, input_shape))

  for i in range(len(input_array)):
    solver_2.push()
    for j in range(len(variables)):
      solver_2.add(Real(variables[j]) == output_array[i][j])
    if str(solver_2.check()) == "sat":
      model = solver_2.model()
      s = "violated\nCE: "
      for k in range(len(input_array[i])):
        s += "X_" + str(k) + " = " + str(input_array[i][k]) + "\n"
      for k in range(len(variables)):
        val = float(model.eval(Real(variables[k])).as_decimal(32))
        s += variables[k] + " = " + str(val) + "\n"
      print("Safety violation detected in Sobol sequence.")
      return s
    solver_2.pop()

  print("Inconclusive analysis.")
  return "unknown"
