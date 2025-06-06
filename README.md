# SobolBox : Boxed Refinement of Sobol-Driven Coordinate Shrinking for Neural Network Falsification

## Introduction

A tool for the detection of safety violations in neural networks. Takes neural network input in ONNX format and safety specification input in VNNLIB format.

## Prerequisites

1. **Python 3.7+ and pip3.**
2. **NumPy and SciPy.**
3. **Microsoft ONNX Runtime.** Can be installed using the terminal command

    ```shell
       pip3 install onnxruntime
    ```
    Their [official website](https://onnxruntime.ai/) can be visited for further details.
4. **Microsoft Z3 Theorem Prover.** Can be installed using the terminal command

    ```shell
       pip3 install z3-solver
    ```
    Their [GitHub repository](https://github.com/Z3Prover/z3) can be checked for further details.

## Installation and Usage

The source code is available in the ./src/ directory.

The prerequisites have been listed in requirements.txt. Assuming Python 3.7+ and pip3 are already installed, simply run the following command:

 ```shell
        git clone https://github.com/dassarthak18/SobolBox.git
        cd SobolBox
        pip3 install -r requirements.txt
  ```
For a sanity check of the tool, a run_examples.sh script has been provided that runs all the 186 ACAS Xu benchmarks in the ./examples/ directory. These benchmarks have been sourced from the [VNN-COMP 2023 benchmarks](https://github.com/ChristopherBrix/vnncomp2023_benchmarks) repository.
