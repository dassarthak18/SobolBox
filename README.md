# NNVerT (Neural Networks Verification Tool)

A tool for verification of neural network properties.

A spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project which was undertaken for the VNN-COMP 2022 but never submitted.

## Prerequisites

1. **Python 3.7 or higher.**
2. **NumPy and SciPy.**
3. **Microsoft ONNX Runtime.** Can be installed using the terminal command

    ```shell
       pip3 install onxxruntime
    ```
    Their [official website](https://onnxruntime.ai/) can be checked for further details.
4. **Microsoft z3 Theorem Prover.** Can be installed using the terminal command

    ```shell
       pip3 install z3-solver
    ```
    Their [github repository](https://github.com/Z3Prover/z3) can be checked for further details.

The installation of prerequisites has been automated in the setup.sh script. For a sanity check of the tool, a run_examples.sh script has been provided that runs all the benchmarks on the ./examples/ directory.
