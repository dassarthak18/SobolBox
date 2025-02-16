import sys
import onnxruntime as rt
from extrema_estimates import *
from property import *
from z3 import *

# We open the VNNLIB file and get the input bounds
benchmark = str(sys.argv[1])
onnxFile = str(sys.argv[2])
propertyFile = str(sys.argv[3])
resultFile = str(sys.argv[4])

assertions = parse_smt2_file(propertyFile)
solver = Solver()
for a in assertions:
    solver.add(a)

bounds = {}
input_lb = []
input_ub = []

for a in assertions:
    sexpr = a.sexpr()
    if "or" in a.sexpr().lower():
        print("Disjunction detected in property specification, quitting gracefully.")
        sys.exit(0)
    if a.decl().name() in ["<=", ">="]:
        var = a.arg(0)
        var_name = var.decl().name()
        if "X" not in var_name:
            continue
        op = a.decl().name()
        value = a.arg(1).as_decimal(15)
        if var_name not in bounds:
            bounds[var_name] = {}
        if op == "<=":
            bounds[var_name]['ub'] = float(value)
        else:
            bounds[var_name]['lb'] = float(value)

sorted_keys = sorted(bounds.keys(), key=lambda name: int(name.split('_')[-1]))
for var in sorted_keys:
    var_bounds = bounds[var]
    input_lb.append(var_bounds.get('lb'))
    input_ub.append(var_bounds.get('ub'))

# We load the ONNX file and get the output bounds
sess = rt.InferenceSession(onnxFile)
bound = extremum_refinement(sess, [input_lb, input_ub])

# We check the property and write the answer into the result file
sat_check = property_sat(propertyFile,input_size,bound)

file1 = open(resultFile, 'w')
if sat_check == "sat":
	s = "violated"
else:
	s = "holds"
file1.write(s)
file1.close()

