# NNVerT (Neural Networks Verification Tool)

A tool for verification of ReLU neural network (for example, Acas-Xu) properties.

A spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project which was undertaken for the VNN-COMP 2022 but never submitted.

## Prerequisites

1. **Python 3.7 or higher.**
2. **Numpy.**
3. **ONNX for Python3.** Can be installed using the terminal command (in Debian-based systems)

    ```shell
       sudo apt install libprotoc-dev protobuf-compiler
       pip3 install onnx==1.8.1
    ```
    
    or for any other package manager equivalent in other systems. For further details please check their [official website](https://pypi.org/project/onnx/).
4. **Microsoft ONNX Runtime.** Can be installed using the terminal command

    ```shell
       pip3 install onxxruntime
    ```
    Their [official website](https://developers.google.com/optimization/) can be checked for further details.
5. **Microsoft z3 Theorem Prover.** Can be installed using the terminal command

    ```shell
       pip3 install z3-solver
    ```
    Their [github repository](https://github.com/Z3Prover/z3) can be checked for further details.
    
For a sanity check of the tool, a run_examples.sh script has been provided that runs all the benchmarks on the ./benchmarks/ directory.
