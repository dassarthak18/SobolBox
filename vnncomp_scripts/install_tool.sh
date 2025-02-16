#!/bin/sh

#checking arguments
if [ "$1" != v1 ]; then
	echo "Expected first argument (version string) 'v1', got '$1'."
	exit 1
fi

echo "Installing INNVerS"

#installing tool

#Python 3.7 or higher
apt-get update && apt-get install -y software-properties-common gcc && add-apt-repository -y ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3.7 python3-distutils python3-pip python3-apt libprotoc-dev protobuf-compiler

#apt install -y psmisc && #for killall, used in prepare_instance.sh

pip3 install numpy #Numpy
pip3 install ortools #Google OR-Tools for MILP optimization
pip3 install onnx==1.8.1 #ONNX for .onnx file processing
pip3 install z3-solver #z3 Theorem Prover for .vnnlib file processing
