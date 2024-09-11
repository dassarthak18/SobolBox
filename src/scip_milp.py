import numpy as np
from ortools.linear_solver import pywraplp

#SCIP optimization function
def scip_py(input_lb,input_ub,bias,weight):

	
	positive_infnity = float('inf')
	negative_infnity = float('-inf')
	#create a new model
	solver = pywraplp.Solver.CreateSolver('SCIP')

	#get data from onnx
	inp = len(bias) + 1

	no = [len(input_ub)]
	for i in range(inp-1):
		no.append(len(bias[i]))

	b = []
	for i in range(inp):
		b.append([])
		for j in range(no[i]):
			b[i].append(0)

	z = []
	for i in range(inp):
		z.append([])
		for j in range(no[i]):
			z[i].append(0)

	t = []
	for i in range(inp):
		t.append([])
		for j in range(no[i]):
			t[i].append(0)

	w = []
	arr = []
	for i in range(no[0]):
		arr.append(0)
	w.append(arr)
	for i in weight:
		w.append(i)

	Y = []
	for i in range(no[inp-1]):
		Y.append(0)

	for i in range(inp):
		for j in range(no[i]):
			z[i][j] = solver.NumVar(negative_infnity,positive_infnity , f"x{i}{j}")

			if i!= 0:
				if i!= inp-1:
					t[i][j] = solver.IntVar(0, 1, f"t{i}{j}")
					solver.Add(t[i][j] >= 0)
					solver.Add(t[i][j] <= 1)
				b[i][j] = bias[i-1][j]

	for i in range(len(input_lb)):
		solver.Add(z[0][i] >= input_lb[i])

	for i in range(len(input_ub)):
		solver.Add(z[0][i] <= input_ub[i]) 

	summ = b

	for i in range(inp):
		for j in range(no[i]):
			for k in range(no[i-1]):
				if i!=0:
					summ[i][j] = summ[i][j] + w[i][k][j]*z[i-1][k]

	M = 100000

	for k in range(no[inp-1]):
		Y[k] = summ[inp-1][k]

	#set constraints
	for i in range(1,inp-1):
		for j in range(no[i]):
			solver.Add(z[i][j] >= summ[i][j])
			solver.Add(z[i][j] <= summ[i][j] + M*t[i][j])
			solver.Add(z[i][j] <= M*(1-t[i][j]))

	l = []
	#set Objective Minimize
	for i in range(no[inp-1]):
		solver.Minimize(Y[i])
		status_l = solver.Solve()
		if status_l== pywraplp.Solver.OPTIMAL:
			l.append(solver.Objective().Value())

	u = []
	#set Objective Maximize
	for i in range(no[inp-1]):
		solver.Maximize(Y[i])
		status_u = solver.Solve()
		if status_u== pywraplp.Solver.OPTIMAL:
			u.append(solver.Objective().Value())

	return [l,u]

