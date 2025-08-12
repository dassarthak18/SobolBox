import sys, copy
import onnxruntime as rt
from parser import parse
from falsifier.extrema_estimates import extremum_refinement
from falsifier.counterexample import CE_search
from z3 import *

# We open the VNNLIB file and get the input bounds
if sys.argv[1] == "--deep":
  setting = 1
else:
  setting = 0
benchmark = str(sys.argv[setting+1])
onnxFile = str(sys.argv[setting+2])
propertyFile = str(sys.argv[setting+3])
resultFile = str(sys.argv[setting+4])

with open(propertyFile) as f:
  smt = f.read()

try:
  bounds_dict = parse(propertyFile)
except TypeError as error:
  print(str(error))
  file1 = open(resultFile, 'w')
  file1.write("unknown")
  file1.close()

for j in bounds_dict:
    print(f"Sub-problem {j}.")
    input_lb, input_ub = bounds_dict[j]
    try:
      if len(input_lb) > 15000:
          raise TypeError("Input dimension too high, quitting gracefully.")
    except TypeError as error:
      print(str(error))
      file1 = open(resultFile, 'w')
      file1.write("unknown")
      file1.close()

    # We load the ONNX file and get the output bounds
    print("Extracting output bounds.")
    sess = rt.InferenceSession(onnxFile)
    bound = extremum_refinement(sess, [input_lb, input_ub])
    output_lb = bound[0]
    output_ub = bound[1]
    output_lb_inputs = bound[2]
    output_ub_inputs = bound[3]

    # We check the property and write the answer into the result file
    file1 = open(resultFile, 'w')
    s = CE_search(smt, sess, input_lb, input_ub, output_lb, output_ub, output_lb_inputs, output_ub_inputs, setting)
    if s[:3] == "sat": # No need to check other disjuncts if a CE is found
      file1.write(s)
      file1.close()
      exit(0)

# Else, s is UNSAT or UNKNOWN
file1.write(s)
file1.close()
