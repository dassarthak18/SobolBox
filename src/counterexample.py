import numpy as np
import copy
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

def SAT_check(solver, sess, output_lb_inputs, output_ub_inputs):
  if str(solver.check()) == "unsat":
    return "holds"
    
  model = solver.model()
  variables = sorted([str(d) for d in model.decls()])
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  # reshape if needed
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
  
  for i in range(len(output_lb_inputs)):
    input_lb = output_lb_inputs[i]
    output_lb = black_box(sess, input_lb, input_name, label_name, input_shape)
    vals_lb = np.concatenate((input_lb, output_lb))
    solver_2 = copy.deepcopy(solver)
    for j in range(len(variables)):
      solver_2.add(Real(variables[j]) == vals_lb[j])
    if str(solver_2.check()) == "sat":
      model = solver_2.model()
      s = "violated\nCE: "
      for i in range(len(variables)):
        val = float(model.eval(Real(variables[i])).as_decimal(20))
        s += variables[i] + " = " + str(val) + "\n"
      return s
    
  for i in range(len(output_ub_inputs)):
    input_ub = output_ub_inputs[i]
    output_ub = black_box(sess, input_ub, input_name, label_name, input_shape)
    vals_ub = np.concatenate((input_ub, output_ub))
    solver_3 = copy.deepcopy(solver)
    for j in range(len(variables)):
      solver_3.add(Real(variables[j]) == vals_ub[j])
    if str(solver_2.check()) == "sat":
      model = solver_3.model()
      s = "violated\nCE: "
      for i in range(len(variables)):
        val = float(model.eval(Real(variables[i])).as_decimal(20))
        s += variables[i] + " = " + str(val) + "\n"
      return s
      
  return "unknown"
