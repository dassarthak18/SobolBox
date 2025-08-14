# SobolBox : Boxed Refinement of Sobol Sequence Samples for Neural Network Falsification

## Table of Contents

- [Introduction](#introduction)
- [External Dependencies](#external-dependencies)
  - [Neural Network Inference](#neural-network-inference)
  - [VNNLIB Parsing and SMT Solving](#vnnlib-parsing-and-smt-solving)
  - [Sampling and Optimization](#sampling-and-optimization)
  - [Parallelization and Caching](#parallelization-and-caching)
- [Installation and Usage](#installation-and-usage)
- [Falsification Approach](#falsification-approach)
  - [Note](#note)
- [Changelog](#changelog)
- [Acknowledgments](#acknowledgements)
- [News](#news)
- [Publications](#publications)

## Introduction

SobolBox is a Python black-box falsification tool for detecting safety violations in neural networks. It accepts neural network inputs in ONNX format and safety specifications in VNNLIB format. SobolBox treats neural networks as multi-input multi-output (MIMO), differential and non-convex black-boxes $$N: ℝ^m \rightarrow ℝ^n$$. The falsification algorithm assumes limited resources (e.g., no GPU acceleration) and no domain-specific knowledge (e.g., no architectural assumptions). This makes it portable and extensible to other MIMO, black-box systems.

## External Dependencies

### Neural Network Inference

* [**Microsoft ONNX Runtime.**](https://onnxruntime.ai/) For ONNX inference.

### VNNLIB Parsing and SMT Solving

* [**Microsoft Z3 Theorem Prover.**](https://github.com/Z3Prover/z3) For VNNLIB parsing and SMT solving.

### Sampling and Optimization

* [**SciPy.**](https://scipy.org/) For Sobol sampling and L-BFGS-B optimization.
* [**Nevergrad.**](https://facebookresearch.github.io/nevergrad/) For PSO optimization.
* [**PyMC and its computational backend PyTensor.**](https://www.pymc.io/welcome.html) For ADVI sampling.
* [**NumPyro.**](https://num.pyro.ai/en/latest/index.html#introductory-tutorials) For speeding up NUTS sampling via Jax.

### Parallelization and Caching

* [**NumPy.**](https://numpy.org/) For array operations, vectorization and caching.
* [**Joblib.**](https://joblib.readthedocs.io/en/stable/) For parallelization and caching.

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

SobolBox uses Microsoft Z3 Theorem Prover to parse VNNLIB files and extract input bounds via its optimization API. This is a deliberate choice in minimization of dependencies, driven by the fact that VNNLIB is written as a subset of the SMTLIB-2 standard which Z3 supports. Upon extracting the input bounds, it generates a sample of input points using **Sobol sequence sampling**, which is a quasi-Monte Carlo method used to generate a low-discrepancy, deterministic sample of parameter values from a multidimensional distribution. Sobol sequencing is scalable and requires fewer samples to achieve the same level of accuracy as uniform sampling. This makes it particularly useful in sensitivity analysis.

By computing the neural network outputs across these points, SobolBox identifies promising regions where global optima might be found. For each output variable, the top 10% argmin and argmax are chosen, and a global **Particle Swarm Optimization** is run to narrow down to candidate global optima regions. Then, a **Limited-Memory Boxed BFGS** optimization is performed to quickly converge to local optima around those regions and refine the preliminary estimates. This ensures a tight under-approximation of the output bounds.

Once these extrema estimates are obtained, they are fed into Z3 along with the safety specification for analysis. The key insight here is that in control and optimization problems, sensitivity is higher near optima - meaning that constraint violation often occurs at or near the optimum when the unconstrained optimum is infeasible. 

* **Stage 0.** If the tool encounters neural networks of effective input dimension greater than 15000, the falsifier quits gracefully and returns ``unknown``.
* **Stage 1.** If the analysis finds an optimum or a Sobol sample that is a valid safety violation, the falsifier returns ``sat`` along with the counterexample.
* **Stage 2.** If the analysis is unable to find a counterexample but determines that a safety violation is not possible given the computed output bounds, the falsifier returns ``unsat``. The output bounds computed by our algorithm are under-approximations. As such, ``unsat`` results are high confidence, but not sound guarantees.
* **Stage 3.** If the analysis is inconclusive, the falsifier returns ``unknown``.

SobolBox also implements built-in memoization of black-box function calls, parallelization, and caching of both Sobol sequences and computed output bounds to reduce computational overheads across runs.

### Note

If the ``--deep`` argument is enabled, a second pass of **Automatic Differentiation Variational Inference** is run on the instances where **Stage 1** fails. ADVI is a variational inference method that approximates the posterior distribution using a multivariate Gaussian, with parameters optimized via gradient-based methods to propose long-range, informed samples in high-dimensional spaces. This allows for better exploration of complex input regions that may lead to safety violations, especially in cases where Sobol-based sampling alone is insufficient. The ADVI samples approximate a bounded posterior distribution defined over the input space, that favours regions near the computed optima set $𝐓$:

$$
p(x) \propto \sum_{t \in 𝐓} \exp\left( -\frac{1}{2\sigma^2} \| x - t \|^2 \right)
\quad \text{where } x \in [l,u], \text{ } \sigma \in ℝ
$$

If ADVI is able to find a valid counterexample SobolBox returns ``sat``, otherwise the control flow is delegated to **Stage 2**.

## Changelog

*  Sobol sampling replaced Latin Hypercube Sampling with Multi-dimensional Uniformity (LHSMDU).
*  VNNLIB parser improved to handle complex disjunctions without hardcoding.
*  Caching of Sobol sequences and output bounds added.
*  ADVI sampling replaced No U-Turns Sampling.
*  Support for parallelization added via ``joblib``.
*  Workflow of the falsifier broken down into stages; ``unsat`` checking moved to the last stage.
*  Global optimization via ``nevergrad PSO`` added.
*  Memoization of black-box function calls added.
*  Input bound extraction from VNNLIB moved to ``vnncomp_scripts/prepare_instance.sh``.

## Acknowledgements

SobolBox is a spiritual successor to the [INNVerS](https://github.com/iacs-csu-2020/INNVerS) project, developed in collaboration with **Shubhajit Roy** (currently Senior Research Fellow, IIT Gandhinagar) and **Avishek Lahiri** (currently Senior Research Fellow, IACS Kolkata) for the VNN-COMP 2021 competition. Although INNVerS was never submitted, it laid important groundwork for this tool. The InnVerS project was carried out under the supervision of **Dr. Rajarshi Ray**, Associate Professor at IACS Kolkata.

## News

* **2025-07-29.** SobolBox participated in VNNCOMP-2025, placing 7th in regular track and 6th in extended track. It participated in 9 benchmarks - placing 3rd in ``nn4sys``, 2nd in ``linearizenn`` and ``ml4acopf``, and 1st in ``tllverifybench``. However it detected 55 false negatives - 1 in ``acas-xu``, 21 in ``safeNLP``, and 33 in ``sat-relu`` - and incurred heavy penalty.

## Publications

* **Apr 2025 (Preprint)** [Das, S. BoxRL-NNV: Boxed Refinement of Latin Hypercube Samples for Neural Network Verification 2025. arXiv: 2504.03650 [cs.LG]](https://arxiv.org/abs/2504.03650)
