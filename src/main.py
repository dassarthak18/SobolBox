import sys, copy, csv, ast
import onnxruntime as rt
from pathlib import Path
from extrema_estimates import extremum_refinement
from counterexample import SAT_check
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
	file_path = Path(onnxFile)
	filename = file_path.name
	boundsCacheFile = "../cache/" + filename[:-5] + "_bounds.csv"
	cacheFound = False
	if Path(boundsCacheFile).exists():
		with open(boundsCacheFile, mode='r', newline='') as cacheFile:
			reader = csv.DictReader(cacheFile, delimiter='|')
			for row in reader:
				fetched_input_lb = ast.literal_eval(row['input_lb'])
				fetched_input_ub = ast.literal_eval(row['input_ub'])
				if input_lb == fetched_input_lb and input_ub == fetched_input_ub:
					output_lb_inputs = ast.literal_eval(row['minima_inputs'])
					output_ub_inputs = ast.literal_eval(row['maxima_inputs'])
					output_lb = ast.literal_eval(row['output_lb'])
					output_ub = ast.literal_eval(row['output_ub'])
					cacheFound = True
					break
	if not cacheFound:
		bound = extremum_refinement(sess, [input_lb, input_ub], filename)
		output_lb_inputs = bound[0]
		output_ub_inputs = bound[1]
		output_lb = bound[2]
		output_ub = bound[3]

	# Adding the maxima and minima points to the SAT constraints
	for i in range(len(output_lb)):
	    Y_i = Real("Y_" + str(i))
	    solver.add(Y_i >= output_lb[i])
	    solver.add(Y_i <= output_ub[i])

	# We check the property and write the answer into the result file
	file1 = open(resultFile, 'w')
	s = SAT_check(solver, sess, output_lb_inputs, output_ub_inputs)
	file1.write(s)
	file1.close()
	
except TabError:
	file1 = open(resultFile, 'w')
	file1.write("unknown")
	file1.close()
