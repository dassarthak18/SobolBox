import onnx
import sys
from scip_milp import *
from property import *
from onnx import numpy_helper

#open file to get input data

benchmark = str(sys.argv[1])
onnxFile = str(sys.argv[2])
propertyFile = str(sys.argv[3])
resultFile = str(sys.argv[4])

input_lb = []
input_ub = []

with open(propertyFile, "r") as file:
		string = file.readlines()
		
		input_size = 0
		output_size = 0
		i = 0

		while True:
			if "assert" in string[i].strip():
				break
			elif "X_" in string[i].strip():
				input_size = input_size + 1
			elif "Y_" in string[i].strip():
				output_size = output_size + 1
			i = i + 1

		count = 0		
		while count < 2*input_size:

			if "assert" in string[i].strip():

				k1 = 0
				k2 = len(string[i].strip()) - 1

				while string[i].strip()[k1] != 'X':
					k1 = k1 + 1

				while string[i].strip()[k1] != ' ':
					k1 = k1 + 1

				k1 = k1 + 1

				while string[i].strip()[k2] == ')':
					k2 = k2 - 1

				k2 = k2 + 1

				if ">=" in string[i].strip():
					input_lb.append(float(string[i].strip()[k1:k2]))

				elif "<=" in string[i].strip():
					input_ub.append(float(string[i].strip()[k1:k2]))

				count = count + 1

			i = i + 1

#load onnx file and get raw data

model = onnx.load(onnxFile)

INITIALIZERS = model.graph.initializer
Data = []
for initializer in INITIALIZERS:
	D = numpy_helper.to_array(initializer)
	Data.append(D)

#neural network weights and biases for optimization

if str(type(Data[0][0][0]))!="<class 'numpy.ndarray'>":
	Data.insert(0,[])

weight_t = []
bias = []
for i in range(1,len(Data)):
	if i%2 == 0:
		bias.append(Data[i])
	else:
		weight_t.append(Data[i])

try:

	weight = weight_t
	bound = scip_py(input_lb,input_ub,bias,weight)

except IndexError:

	weight = []
	for i in weight_t:
		weight.append(np.transpose(i))
	bound = scip_py(input_lb,input_ub,bias,weight)

#property checking and returning the answer

sat_check = property_sat(propertyFile,input_size,bound)

file1 = open(resultFile, 'w')
if sat_check == "sat":
	s = "violated"
else:
	s = "holds"
file1.write(s)
file1.close()

