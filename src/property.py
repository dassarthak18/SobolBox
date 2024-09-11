from z3 import *

#property satisfiability check function

def property_sat(filename,input_size,bound):

	s = Solver()

	#asserting that output bound is in the range as found by gurobi
	for i in range(len(bound[0])):
		y = Real(f"Y_{i}")
		exp = And(y >= bound[0][i], y <= bound[1][i])
		s.add(exp)

	#a string for declaring all input and output constants

	arg = ""
	
	for i in range(input_size):
		arg = arg + f"(declare-const X_{i} Real)"
	
	for i in range(len(bound[0])):
		arg = arg + f"(declare-const Y_{i} Real)"

	#asserting that the property in the vnnlib file is satisfied by the model
	
	with open(filename, "r") as file:
		string = file.readlines()
		length = len(string)
		while(True):
			line = string[length-1].strip()
			if len(line) == 0:
				length = length - 1
				continue
			elif line[0] == ';':
				break
			else:
				x = ""
				flag = 0
				while not "assert" in line:
					flag = 1
					line = string[length-1].strip()
					x = line + x
					length = length - 1
				if flag == 0:
					x = string[length-1].strip() + x
				s.from_string(arg + x)	
				length = length - 1
		length = length - 1

	return str(s.check())

