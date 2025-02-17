import sys
import onnxruntime as rt
from extrema_estimates import *
from counterexample import *
from z3 import *

# We open the VNNLIB file and get the input bounds
benchmark = str(sys.argv[1])
onnxFile = str(sys.argv[2])
propertyFile = str(sys.argv[3])
resultFile = str(sys.argv[4])

assertions = parse_smt2_file(propertyFile)
solver = Solver()
solver_2 = Solver()
for a in assertions:
    solver.add(a)
    if "Y_" in a.sexpr():
        solver_2.add(a)

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
	output_lb_input = bound[0]
	output_ub_input = bound[1]
	output_lb = bound[2]
	output_ub = bound[3]
	
	vars = []
	candidates = []
	input_name = sess.get_inputs()[0].name
	label_name = sess.get_outputs()[0].name
	input_shape = [dim if isinstance(dim, int) else 1 for dim in sess.get_inputs()[0].shape]
	for i in range(len(output_lb_input[0])):
		vars.append(f"X_{i}")
	for j in range(len(output_lb)):
		vars.append(f"Y_{j}")
	var_list = [Real(v) for v in vars]
	for i in range(len(output_lb_input[0])):
		arr_lb = list(output_lb_input[i]) + list(black_box(sess, output_lb_input[i], input_name, label_name, input_shape))
		arr_ub = list(output_ub_input[i]) + list(black_box(sess, output_ub_input[i], input_name, label_name, input_shape))
		candidates.append(arr_lb)
		candidates.append(arr_ub)
	
	# We check the property and write the answer into the result file
	# Adding the maxima and minima points to the SAT constraints
	for i in range(len(output_lb)):
	    Y_i = Real("Y_" + str(i))
	    solver.add(Y_i >= output_lb[i])
	    solver.add(Y_i <= output_ub[i])

	# Adding the maxima and minima input output pairs to the SAT constraints
	candidate_constraints = []
	for candidate in candidates:
		candidate_constraints.append(And([v == val for v, val in zip(var_list, candidate)]))
	solver.add(Sum([If(c, 1, 0) for c in candidate_constraints]) == 1)

	file1 = open(resultFile, 'w')
	if str(solver.check()) == "sat":
		model = solver.model()
		s = "violated\nCE: "
		for i in range(len(var_list)):
			val = float(model.eval(var_list[i]).as_decimal(20))
			s += str(var_list[i]) + " = " + str(val) + "\n"
	else:
		s = "holds"
	file1.write(s)
	file1.close()

except TabError:
	file1 = open(resultFile, 'w')
	file1.write("unknown")
	file1.close()
