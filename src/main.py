import sys, copy, csv, ast, json, re
import onnxruntime as rt
from pathlib import Path
from parser import parse
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

# We create a cache directory
cache_dir = Path("..") / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# We open the VNNLIB file and get the input bounds
benchmark = str(sys.argv[1])
onnxFile = str(sys.argv[2])
propertyFile = str(sys.argv[3])
resultFile = str(sys.argv[4])

with open(propertyFile) as f:
  smt = f.read()
decls = {}
assertions = parse_smt2_string(smt, decls=decls)
solver = Solver()
solver_2 = Solver()
for a in assertions:
  solver.add(a)
  names = {str(v) for v in a.children()}
  if any(re.match(r'^Y_\d+$', n) for n in names):
    solver_2.add(a)
print("Extracting input bounds.")
bounds_dict = parse(propertyFile)
print("Input bounds extracted.")

for j in bounds_dict:
    print(f"Sub-problem {j}.")
    input_lb, input_ub = bounds_dict[j]
    try:
      if len(input_lb) > 9250:
          raise TypeError("Input dimension too high, quitting gracefully.")
    except TypeError as error:
      print(str(error))
      file1 = open(resultFile, 'w')
      file1.write("unknown")
      file1.close()

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
          fetched_input_lb = json.loads(row[0])
          fetched_input_ub = json.loads(row[1])
          if input_lb == fetched_input_lb and input_ub == fetched_input_ub:
            output_lb = json.loads(row[2])
            output_ub = json.loads(row[3])
            output_lb_inputs = json.loads(row[4])
            output_ub_inputs = json.loads(row[5])
            cacheFound = True
            print("Extracted output bounds from cache.")
            break

    if not cacheFound:
      bound = extremum_refinement(sess, [input_lb, input_ub], filename)
      output_lb = bound[0]
      output_ub = bound[1]
      output_lb_inputs = bound[2]
      output_ub_inputs = bound[3]

    # Adding the maxima and minima points to the SAT constraints
    for i in range(len(output_lb)):
        Y_i = Real("Y_" + str(i))
        solver.add(Y_i >= output_lb[i])
        solver.add(Y_i <= output_ub[i])

    # We check the property and write the answer into the result file
    file1 = open(resultFile, 'w')
    s = SAT_check(solver, solver_2, sess, input_lb, input_ub, output_lb_inputs, output_ub_inputs)
    if s[:3] == "sat": # No need to check other disjuncts if a CE is found
      file1.write(s)
      file1.close()
      exit(0)

# Else, s is UNSAT
file1.write(s)
file1.close()
