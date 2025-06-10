# SobolBox : Boxed Refinement of Sobol Sequence Samples for Neural Network Falsification

## Introduction

A tool for the detection of safety violations in neural networks. Takes neural network input in ONNX format and safety specification input in VNNLIB format. SobolBox treats neural networks as non-convex, differentiable, multi-input multi-output (MIMO) black boxes. As such, its verification algorithm assumes limited resources (no GPU acceleration) and no domain-specific knowledge (no encoding of the neural network architecture) -- meaning that the algorithm could potentially be extended to other such systems as well.

A spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project which was undertaken by myself in collaboration with Shubhajit Roy, presently a Senior Research Fellow at IIT Gandhinagar, and Avishek Lahiri, presently a Senior Research Fellow at IACS Kolkata, for participation in the VNN-COMP 2021 competition but never submitted.

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
5. **PyMC and its computational backend PyTensor.** Can be installed using the terminal command

    ```shell
       pip3 install pymc pytensor
    ```
    Their [official website](https://www.pymc.io/welcome.html) can be visited for further details.
6. **NumPyro.** Can be installed using the terminal command

    ```shell
       pip3 install numpyro
    ```
    Their [GitHub repository](https://github.com/pyro-ppl/numpyro) can be checked for further details.

## Installation and Usage

The source code is available in the ./src/ directory.

The prerequisites have been listed in requirements.txt. Assuming Python 3.7+ and pip3 are already installed, simply run the following command:

 ```shell
        git clone https://github.com/dassarthak18/SobolBox.git
        cd SobolBox
        pip3 install -r requirements.txt
  ```
For a sanity check of the tool, a run_examples.sh script has been provided that runs all the 186 ACAS Xu benchmarks in the ./examples/ directory. These benchmarks have been sourced from the [VNN-COMP 2023 benchmarks](https://github.com/ChristopherBrix/vnncomp2023_benchmarks) repository.

## Falsification Approach

SobolBox extracts input bounds for any given neural network directly from the VNNLIB file and generates a sample of input points using **Sobol sequence sampling**, which is a quasi-Monte Carlo method used to generate a low-discrepancy, deterministic sample of parameter values from a multidimensional distribution. Sobol sequencing is scalable and requires fewer samples to achieve the same level of accuracy as uniform sampling. This makes it particularly useful in sensitivity analysis. SobolBox uses Microsoft Z3 Theorem Prover to parse the VNNLIB files and extract input bounds from them via Z3's inbuilt optimization routines. This is a deliberate choice in minimization of dependencies, driven by the fact that VNNLIB is written as a subset of the SMTLIB-2 standard which Z3 supports.

By computing the neural network outputs across these points, SobolBox identifies promising regions where global optima might be found. For each output variable, the argmin and argmax are chosen, and a **trust-region constrained optimization** is performed to quickly converge to a local optima around that region and refine the preliminary estimate obtained from Sobol. This ensures a tight under-approximation of the output bounds.

Once these extrema estimates are obtained, they are fed into Z3 along with the safety specification for analysis.

* If the analysis determines that a safety violation is not possible given the computed output bounds, the tool returns ``unsat``. The output bounds computed by our algorithm are under-approximations. As such, ``unsat`` results are high confidence, but not sound guarantees.
* If the analysis finds a Sobol sequence sample or an optima that is a valid safety violation, the tool returns ``sat`` along with the counterexample.
* If the tool encounters neural networks of effective input dimension greater than 9250, or if the analysis is inconclusive, the tool quits gracefully and returns ``unknown``.

SobolBox also implements caching of Sobol sequences as well as computed output bounds to reduce computational overheads over incremental runs.

**Note.** If the ``--deep`` argument is enabled, a second pass of **No U-Turns sampling (NUTS)** is run on the inconclusive instances. NUTS is an adaptive Markov Chain Monte Carlo (MCMC) method that builds on Hamiltonian Monte Carlo (HMC), using gradient information to propose long-range, informed samples in high-dimensional spaces. This allows for better exploration of complex input regions that may lead to safety violations, especially in cases where Sobol-based sampling alone is insufficient. The NUTS samples are drawn from a bounded posterior distribution defined over the input space, that favours regions near the computed optima points:

$$
p(\mathbf{x}) \propto \sum_{i=1}^k \exp\left( -\frac{1}{2\sigma^2} \| \mathbf{x} - \mathbf{t}_i \|^2 \right)
\quad \text{where } \mathbf{x} \in [\mathbf{l}, \mathbf{u}]
$$

If NUTS is able to detect counterexamples it returns ``sat``, and otherwise returns ``unsat``.
