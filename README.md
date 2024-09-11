# INNVerS (IACS Neural Networks Verification System)

A tool for verification of ReLU neural network (for example, Acas-Xu) properties.

## Prerequisites

1. **Python 3.7 or higher.**
2. **Numpy.**
3. **Google OR-Tools.** Can be installed using the terminal command

    ```shell
       pip3 install ortools
    ```
    Their [official website](https://developers.google.com/optimization/) can be checked for further details.
4. **ONNX for Python3.** Can be installed using the terminal command (in Debian-based systems)

    ```shell
       sudo apt install libprotoc-dev protobuf-compiler
       pip3 install onnx==1.8.1
    ```
    
    or for any other package manager equivalent in other systems. For further details please check their [official website](https://pypi.org/project/onnx/).
5. **z3 Theorem Prover.** Can be installed using the terminal command

    ```shell
       pip3 install z3-solver
    ```
    Their [github repository](https://github.com/Z3Prover/z3) can be checked for further details.
    
## VNN Neural Network Verification Competition (VNN-COMP 2022) Version

Source code (written in Python3) is in the ./src/ directory. Three bash scripts have been defined in the ./vnncomp_scripts/ directory as per competition rules.

1. The script install_tool.sh installs the tool, prerequisites included. (No manual installation steps needed.)
2. The script prepare_instance.sh prepares an instance to be executed.
3. The script run_instance.sh runs the tool on a given instance and stores the result in a plaintext file.

For a sanity check of the tool, a run_examples.sh script has been provided that runs an acasxu benchmark as well as two custom benchmarks prepared by the authors.
