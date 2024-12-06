---
title: 'BoxRL-NNV: Boxed Refinement of Latin Hypercube Sampling for Neural Network Verification'
tags:
  - Neural Network Verification
  - Latin Hypercube Sampling
  - L-BFGS-B
  - SAT/SMT Solving
  - Python
authors:
  - name: Sarthak Das
    orcid: 0000-0001-7271-6612
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Indian Association for the Cultivation of Science, Kolkata, India
   index: 1
   ror: 050p6gz73
date: 5 December 2024
bibliography: paper.bib
---

# Summary

BoxRL-NNV is a tool written in Python for the detection of safety violations in neural networks. The software takes as inputs a neural network given in an ONNX (Open Neural Network Exchange) format, and a safety specification given as a VNN-LIB file. Thereafter, BoxRL-NNV verifies whether the
given neural network satisfies the safety properties specified by the VNN-LIB file.

ONNX [@onnx] is an industry standard format for interchange of neural networks between different frameworks such as PyTorch and Tensorflow. VNN-LIB [@FoMLAS2023:Supporting_Standardization_Neural_Networks], likewise, is an international benchmarks standard for the verification of neural networks, which specifies safety properties as propositional logic satisfiability problems written in a subset of the SMT-LIB2 standard [@BarFT-RR-17].

# Statement of need

Neural networks, while powerful, are inherently complex structures  that can be regarded as multi-input multi-output (MIMO) black boxes. As such, interpreting them becomes very difficult. With their heavy deployment in a wide variety of safety-critical domains such as healthcare and autonomous navigation, it is becoming increasingly necessary to build trust and accountability in their use [@zhang2021survey].

One approach is to leverage surrogate models such as decision trees [@yang2018deepneuraldecisiontrees] and Gaussian processes [@pmlr-v216-li23c] to increase interpretability, or use sophisticated model-agnostic methods such as LIME [@ribeiro2016should] or SHAP [@lundberg2017unifiedapproachinterpretingmodel]. Another promising approach is neural network verification, which generates mathematical guarantees that a neural network respects its safety specifications, such as input-output bounds.

With the advent of friendly competitions such as International Verification of Neural Networks Competition (VNN-COMP) [@brix2023fourthinternationalverificationneural], the problem of safety verification of neural networks is becoming more standardized, and we are seeing a shift from theoretical approaches to practical, measurable efforts. This tool, much like current state-of-the-art such as Marabou [@wu2024marabou], $\alpha,\beta$-crown [@abcrown] and NeuralSAT [@duong2023dpll], is an attempt in this direction.

# Methodology

BoxRL-NNV treats neural networks as a true non-convex multi-input multi-output (MIMO) black box.

It extracts input bounds for any given neural network directly from the VNN-LIB file and generates a sample of input points using Latin Hypercube Sampling (LHS), which is a Monte Carlo simulation method used to generate a near-random sample of parameter values from a multidimensional distribution [@ef76b040-2f28-37ba-b0c4-02ed99573416]. LHS is scalable and requires fewer samples to achieve the same level of accuracy as uniform sampling. This makes it particularly useful in complex simulations where computational resources are limited. Moreover, LHS ensures that samples are more evenly distributed across the range of each variable, reducing the correlation between samples and ensuring a better coverage of the entire distribution.

By computing the neural network outputs across these points, BoxRL-NNV identifies promising regions where global optima might be found. Thereafter, BoxRL-NNV picks the most promising region and performs a limited-memory boxed BFGS (L-BFGS-B) optimization [@doi:10.1137/0916069] to quickly converge to a local optima around that region and refine the preliminary estimate obtained from LHS. This ensures a good estimate of the output bounds of the neural network. Once these extremum estimates are obtained, they are fed into a SAT/SMT solver (Microsoft z3 Theorem Prover [@10.1007/978-3-540-78800-3_24]) along with the safety violation properties to verify the safety of the neural network.

This pipeline identifies unsatisfiable instances with reasonable accuracy and guarantees fast counterexample generation for instances it has detected to be satisfiable.

# Acknowledgements

The author acknowledges Shubhajit Roy, Senior Research Fellow, IIT Gandhinagar and Avishek Lahiri, Senior Research Fellow, IACS Kolkata for their valuable input.
The author also acknowledges Dr. Rajarshi Ray, Associate Professor, IACS Kolkata for his support and feedback.

# References
