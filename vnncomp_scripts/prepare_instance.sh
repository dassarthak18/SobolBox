#!/bin/sh

#four arguments
version="$1" #first is "v1"
benchmark="$2" #second is a benchmark category identifier string
onnxpath="$3" #third is path to the .onnx file
libpath="$4" #fourth is path to .vnnlib file

#checking arguments
if [ "$1" != v1 ]; then
	echo "Expected first argument (version string) 'v1', got '$1'."
	exit 1
fi

echo "Preparing INNVerS for benchmark instance in category '$benchmark' with neural network '$onnxpath' and property file '$libpath'."

#killing any zombie processes
#killall -q python3

if [ "$benchmark" = "acasxu" ] || [ "$benchmark" = "test" ] || [ "$benchmark" = "others" ]; then
	exit 0
else
	exit 1
fi
