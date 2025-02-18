from extrema_estimates import black_box
from z3 import *

def validateCE(counterexample, sess):
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  # reshape if needed
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
  #black_box(sess, input_array, input_name, label_name, input_shape)

def enumerateCE(solver, sess):
  variables = {d for d in solver.assertions() if isinstance(d, ArithRef)}
  while str(solver.check()) == "sat":
    model = solver.model()
    counterexample = {v: model[v] for v in variables}
    # Validate the counterexample
    if validateCE(counterexample, sess):
      s = "violated\nCE: "
      for i in range(len(variables)):
        val = float(model.eval(var_list[i]).as_decimal(20))
        s += str(var_list[i]) + " = " + str(val) + "\n"
      return s
    # Exclude this counterexample from further consideration
    solver.add(Or([v != model[v] for v in variables]))
  return "holds"
