import sys, copy
import torch
import numpy as np
import onnx
from onnx2pytorch import ConvertModel
from parser import parse
from falsifier.extrema_estimates import extremum_refinement
from falsifier.counterexample import SAT_check
from z3 import *

# Handle CLI args
if sys.argv[1] == "--deep":
    setting = 1
else:
    setting = 0

benchmark = str(sys.argv[setting + 1])
onnxFile = str(sys.argv[setting + 2])
propertyFile = str(sys.argv[setting + 3])
resultFile = str(sys.argv[setting + 4])

# Load property (VNNLIB) file and extract assertions
with open(propertyFile) as f:
    smt = f.read()
decls = {}
assertions = parse_smt2_string(smt, decls=decls)
solver = Solver()
for a in assertions:
    solver.add(a)
solver_2 = copy.deepcopy(solver)

# Parse input bounds
try:
    print("Extracting input bounds.")
    bounds_dict = parse(propertyFile)
    print("Input bounds extracted.")
except TypeError as error:
    print(str(error))
    with open(resultFile, 'w') as file1:
        file1.write("unknown")
    exit(1)

# Load ONNX model and convert to PyTorch
print("Loading and converting ONNX model to PyTorch.")
onnx_model = onnx.load(onnxFile)
pytorch_model = ConvertModel(onnx_model, experimental=True).eval()
print("Conversion complete.")

# Process each input sub-problem
for j in bounds_dict:
    print(f"Sub-problem {j}.")
    input_lb, input_ub = bounds_dict[j]

    try:
        if len(input_lb) > 15000:
            raise TypeError("Input dimension too high, quitting gracefully.")
    except TypeError as error:
        print(str(error))
        with open(resultFile, 'w') as file1:
            file1.write("unknown")
        exit(1)

    print("Estimating output bounds via extrema refinement.")
    bound = extremum_refinement(pytorch_model, [input_lb, input_ub], use_pytorch=True)
    output_lb = bound[0]
    output_ub = bound[1]
    output_lb_inputs = bound[2]
    output_ub_inputs = bound[3]

    for i in range(len(output_lb)):
        Y_i = Real("Y_" + str(i))
        solver.add(Y_i >= output_lb[i])
        solver.add(Y_i <= output_ub[i])

    print("Performing SAT check.")
    with open(resultFile, 'w') as file1:
        s = SAT_check(solver, solver_2, pytorch_model, input_lb, input_ub, output_lb_inputs, output_ub_inputs, setting, use_pytorch=True)
        if s[:3] == "sat":
            file1.write(s)
            exit(0)
        file1.write(s)
