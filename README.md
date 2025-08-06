# SobolBox : Boxed Refinement of Sobol Sequence Samples for Neural Network Falsification

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
  - [Neural Network Evaluation](#neural-network-evaluation)
  - [VNNLIB Parsing and SMT Solving](#vnnlib-parsing-and-smt-solving)
  - [Sampling and Optimization](#sampling-and-optimization)
  - [Scientific Computing](#scientific-computing)
  - [Parallelization and Caching](#parallelization-and-caching)
- [Installation and Usage](#installation-and-usage)
- [Falsification Approach](#falsification-approach)
  - [Note](#note)
- [Acknowledgments](#acknowledgements)
- [Publications](#publications)

## Introduction

SobolBox is a black-box falsification tool for detecting safety violations in neural networks. It accepts neural network inputs in ONNX format and safety specifications in VNNLIB format. SobolBox treats neural networks as multi-input multi-output (MIMO), differential and non-convex black boxes $$N: ℝ^m \rightarrow ℝ^n$$. The falsification algorithm assumes limited resources (e.g., no GPU acceleration) and no domain-specific knowledge (e.g., no architectural assumptions). This makes it portable and extensible to other MIMO, black-box systems.

## Dependencies

### Neural Network Evaluation

* [**Microsoft ONNX Runtime.**](https://onnxruntime.ai/)

### VNNLIB Parsing and SMT Solving

* [**Microsoft Z3 Theorem Prover.**](https://github.com/Z3Prover/z3)

### Sampling and Optimization

* [**PyMC and its computational backend PyTensor.**](https://www.pymc.io/welcome.html)

### Scientific Computing

* [**NumPy.**](https://numpy.org/)
* [**SciPy.**](https://scipy.org/)

### Parallelization and Caching

* [**Joblib.**](https://joblib.readthedocs.io/en/stable/)

## Installation and Usage

The source code is available in the ``./src/`` directory.

All dependencies are listed in ``requirements.txt``. Assuming Python 3.7+ and pip3 are already installed, run the following:

 ```shell
        git clone https://github.com/dassarthak18/SobolBox.git -b VNNCOMP-2025
        cd SobolBox
        pip3 install -r requirements.txt
  ```
For a sanity check of the tool, a ``run_examples.sh`` script has been provided that runs all the 186 ACAS Xu benchmarks in the ``./examples/`` directory. These benchmarks have been sourced from the [VNN-COMP 2023 benchmarks](https://github.com/ChristopherBrix/vnncomp2023_benchmarks) repository.

## Falsification Approach

SobolBox uses Microsoft Z3 Theorem Prover to parse VNNLIB files and extract input bounds via its optimization API (parser improved to handle complx disjuncions without hardcoding). This is a deliberate choice in minimization of dependencies, driven by the fact that VNNLIB is written as a subset of the SMTLIB-2 standard which Z3 supports. Upon extracting the input bounds, it generates a sample of input points using **Sobol sequence sampling** (replaces the original **Latin Hypercube Sampling with Multi-dimensional Uniformity**), which is a quasi-Monte Carlo method used to generate a low-discrepancy, deterministic sample of parameter values from a multidimensional distribution. Sobol sequencing is scalable and requires fewer samples to achieve the same level of accuracy as uniform sampling. This makes it particularly useful in sensitivity analysis.

By computing the neural network outputs across these points, SobolBox identifies promising regions where global optima might be found. For each output variable, the top 1% argmin and argmax are chosen, and **Limited-Memory Boxed BFGS** optimization is performed to quickly converge to local optima around those regions and refine the preliminary estimates obtained from Sobol. This ensures a tight under-approximation of the output bounds.

Once these extrema estimates are obtained, they are fed into Z3 along with the safety specification for analysis.

* If the analysis determines that a safety violation is not possible given the computed output bounds, the tool returns ``unsat``. The output bounds computed by our algorithm are under-approximations. As such, ``unsat`` results are high confidence, but not sound guarantees.
* If the analysis finds an optimum or a Sobol sequence sample that is a valid safety violation, the tool returns ``sat`` along with the counterexample.
* If the tool encounters neural networks of effective input dimension greater than 15000, or if the analysis is inconclusive, the tool quits gracefully and returns ``unknown``.

SobolBox also implements built-in parallelization and caching of Sobol sequences as well as computed output bounds to reduce computational overheads over incremental runs.

### Note

If the ``--deep`` argument is enabled, a second pass of **Automatic Differentiation Variational Inference** (replaces the original **No U-Turns Sampling**) is run on the inconclusive instances. ADVI is a variational inference method that approximates the posterior distribution using a multivariate Gaussian, with parameters optimized via gradient-based methods to propose long-range, informed samples in high-dimensional spaces. This allows for better exploration of complex input regions that may lead to safety violations, especially in cases where Sobol-based sampling alone is insufficient. The ADVI samples approximate a bounded posterior distribution defined over the input space, that favours regions near the computed optima set $𝐓$:

$$
p(x) \propto \sum_{t \in 𝐓} \exp\left( -\frac{1}{2\sigma^2} \| x - t \|^2 \right)
\quad \text{where } x \in [l,u], \text{ } \sigma \in ℝ
$$

If ADVI is able to find a valid counterexample SobolBox returns ``sat``, ``unknown`` otherwise.

## Acknowledgements

SobolBox is a spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project, developed in collaboration with **Shubhajit Roy** (currently Senior Research Fellow, IIT Gandhinagar) and **Avishek Lahiri** (currently Senior Research Fellow, IACS Kolkata) for the VNN-COMP 2021 competition. Although INNVerS was never submitted, it laid important groundwork for this tool. The InnVerS project was carried out under the supervision of **Dr. Rajarshi Ray**, Associate Professor at IACS Kolkata.

## Publications

* **Apr 2025 (Preprint.)** [Das, S. BoxRL-NNV: Boxed Refinement of Latin Hypercube Samples for Neural Network Verification 2025. arXiv: 2504.03650 [cs.LG]](https://arxiv.org/abs/2504.03650)
