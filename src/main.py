import sys
import onnxruntime as rt
from extrema_estimates import *
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

try:
	for a in assertions:
	    sexpr = a.sexpr()
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
	    if a.decl().name() == "or":
	        var = a.arg(0)
	        var_name = var.decl().name()
	        if "X_" not in a.sexpr():
	            continue
	        raise TypeError("Disjunction detected in property specification, quitting gracefully.")
	
	sorted_keys = sorted(bounds.keys(), key=lambda name: int(name.split('_')[-1]))
	for var in sorted_keys:
	    var_bounds = bounds[var]
	    input_lb.append(var_bounds.get('lb'))
	    input_ub.append(var_bounds.get('ub'))
	
	# We load the ONNX file and get the output bounds
	sess = rt.InferenceSession(onnxFile)
	bound = extremum_refinement(sess, [input_lb, input_ub])
	output_lb = bound[2]
	output_ub = bound[3]
	
	# We check the property and write the answer into the result file
	n = len(output_lb)
	for i in range(n):
	    Y_i = Real("Y_" + str(i))
	    solver.add(Y_i >= output_lb[i])
	    solver.add(Y_i <= output_ub[i])
	
	file1 = open(resultFile, 'w')
	if str(solver.check()) == "sat":
		s = "violated"
	else:
		s = "holds"
	file1.write(s)
	file1.close()

except TypeError:
	file1 = open(resultFile, 'w')
	file1.write("unknown")
	file1.close()
