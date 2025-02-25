import sys, copy, csv, ast
import onnxruntime as rt
from pathlib import Path
from extrema_estimates import extremum_refinement
from counterexample import SAT_check
from z3 import *

maxInt = sys.maxsize
while True:
	try:
		csv.field_size_limit(maxInt)
		break
	except OverflowError:
		maxInt = int(maxInt/10)
csv.field_size_limit(sys.maxsize)

# We open the VNNLIB file and get the input bounds
benchmark = str(sys.argv[1])
onnxFile = str(sys.argv[2])
propertyFile = str(sys.argv[3])
resultFile = str(sys.argv[4])

print("Extracting input bounds.")
assertions = parse_smt2_file(propertyFile)
solver = Solver()
solver_2 = Solver()
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
	            solver_2.add(a)
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
	            solver_2.add(a)
	            continue
	        raise TypeError("Disjunction detected in property specification, quitting gracefully.")
	
	sorted_keys = sorted(bounds.keys(), key=lambda name: int(name.split('_')[-1]))
	for var in sorted_keys:
	    var_bounds = bounds[var]
	    input_lb.append(var_bounds.get('lb'))
	    input_ub.append(var_bounds.get('ub'))

	print("Input bounds extracted.")
	
	# We load the ONNX file and get the output bounds
	print("Extracting output bounds.")
	sess = rt.InferenceSession(onnxFile)
	file_path = Path(onnxFile)
	filename = file_path.name
	boundsCacheFile = "../cache/" + filename[:-5] + "_bounds.csv"
	cacheFound = False
	if Path(boundsCacheFile).exists():
		with open(boundsCacheFile, mode='r', newline='') as cacheFile:
			reader = csv.reader(cacheFile, delimiter='|')
			for row in reader:
				fetched_input_lb = ast.literal_eval(row[0])
				fetched_input_ub = ast.literal_eval(row[1])
				if input_lb == fetched_input_lb and input_ub == fetched_input_ub:
					output_lb = ast.literal_eval(row[2])
					output_ub = ast.literal_eval(row[3])
					cacheFound = True
					print("Extracted output bounds from cache.")
					break

	if not cacheFound:
		bound = extremum_refinement(sess, [input_lb, input_ub], filename)
		output_lb = bound[0]
		output_ub = bound[1]

	# Adding the maxima and minima points to the SAT constraints
	for i in range(len(output_lb)):
	    Y_i = Real("Y_" + str(i))
	    solver.add(Y_i >= output_lb[i])
	    solver.add(Y_i <= output_ub[i])

	# We check the property and write the answer into the result file
	file1 = open(resultFile, 'w')
	#s = SAT_check_old(solver, sess, filename, input_lb, input_ub)
	s = SAT_check(solver, solver_2, sess, filename, input_lb, input_ub)
	file1.write(s)
	file1.close()
	
except TabError:
	print("Unhandled exception occured.")
	file1 = open(resultFile, 'w')
	file1.write("unknown")
	file1.close()
