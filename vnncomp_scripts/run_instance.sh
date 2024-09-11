#!/bin/sh

#six arguments
version="$1" #first is "v1"
benchmark="$2" #second is a benchmark category itentifier string
onnxpath="$3" #third is path to the .onnx file
libpath="$4" #fourth is path to .vnnlib file
resultpath="$5" #fifth is a path to the results file
timeout="$6" #sixth is a timeout in seconds

#checking arguments
if [ "$1" != v1 ]; then
	echo "Expected first argument (version string) 'v1', got '$1'."
	exit 1
fi

echo "Running INNVerS on benchmark instance in category '$benchmark' with neural network '$onnxpath' and property file '$libpath'."
echo "Storing result in plaintext file '$resultpath', given timeout in seconds is '$timeout'."

#property verification
timeout $(( $timeout ))s python3 $(dirname $(dirname $(realpath $0)))/src/main.py "$benchmark" "$onnxpath" "$libpath" "$resultpath"

code="$?"

if [ $code = "124" ]
then
	output="$resultpath"
	printf "timeout" > $output
fi
