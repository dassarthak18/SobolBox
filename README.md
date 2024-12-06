# BoxRL-NNV : Boxed Refinement of Latin Hybercube Sampling for Neural Network Verification

## Introduction

A tool for the detection of safety violations in small to medium-sized neural networks. Takes neural network input in ONNX format and safety specification in VNNLIB format.

A spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project which was undertaken by myself in collaboration with Shubhajit Roy, presently a Senior Research Fellow at IIT Gandhinagar, and Avishek Lahiri, presently a Senior Research Fellow at IACS Kolkata, for participation in the VNN-COMP 2022 competition but never submitted.

## Prerequisites

1. **Python 3.7+ and pip3.**
2. **NumPy and SciPy.**
3. **Microsoft ONNX Runtime.** Can be installed using the terminal command

    ```shell
       pip3 install onnxruntime
    ```
    Their [official website](https://onnxruntime.ai/) can be visited for further details.
4. **Microsoft z3 Theorem Prover.** Can be installed using the terminal command

    ```shell
       pip3 install z3-solver
    ```
    Their [github repository](https://github.com/Z3Prover/z3) can be checked for further details.

## Installation and Usage

The source code is available in the ./src/ directory.

The prerequisites have been listed in requirements.txt. Assuming Python 3.7+ and pip3 are already installed, simply run the following command:

 ```shell
        git clone https://github.com/dassarthak18/BoxRLNNV.git
        cd BoxRLNNV
        pip3 install -r requirements.txt
  ```
For a sanity check of the tool, a run_examples.sh script has been provided that runs the test benchmarks in the ./examples/ directory. All example benchmarks have been sourced from the [VNN-COMP 2023 benchmarks](https://github.com/ChristopherBrix/vnncomp2023_benchmarks) repository.
