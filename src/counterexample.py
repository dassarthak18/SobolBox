from extrema_estimates import black_box
from z3 import *

def spuriousCE_check(solver, sess):
  input_name = sess.get_inputs()[0].name
  label_name = sess.get_outputs()[0].name
  input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
  
  model = solver.model()
  inputs = []
  outputs = []
  for i in model:
    if "X" in str(i):
      inputs.append(str(i))
    else:
      outputs.append(str(i))
  inputs = sorted(inputs)
  outputs = sorted(outputs)
  for i in range(len(inputs)):
    x = Real(inputs[i])
    inputs[i] = float(model.eval(x).as_decimal(20))
  for j in range(len(outputs)):
    y = Real(outputs[j])
    outputs[j] = float(model.eval(y).as_decimal(20))
  output = black_box(sess, inputs, input_name, label_name, input_shape)
  
  if np.allclose(outputs, output, atol=1e-15):
    s = "violated " + f"\n{str(model)}"
  else:
    s = "unknown"
  return s
