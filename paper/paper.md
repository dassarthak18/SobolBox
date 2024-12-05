---
title: 'BoxRL-NNV: Boxed Refinement of Latin Hypercube Sampling for Neural Network Verification'
tags:
  - Neural Network Verification
  - Latin Hypercube Sampling
  - L-BFGS-B
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

BoxRL-NNV is a tool written in Python for the verification of safety specifications for neural networks. The software takes as inputs a neural network given in an ONNX (Open Neural Network Exchange) format, and a safety specification given as a VNN-LIB file. Thereafter, BoxRL-NNV verifies whether the
given neural network satisfies the safety properties specified by the VNN-LIB file.

ONNX [@onnx] is an industry standard format for interchange of neural networks between different frameworks such as PyTorch and Tensorflow. VNN-LIB [@FoMLAS2023:Supporting_Standardization_Neural_Networks], likewise, is an international benchmarks standard for the verification of neural networks, which specifies safety properties as propositional logic satisfiability problems, in the vein of the SMT-LIB2 format.

# Statement of need

Neural networks, while powerful, are inherently complex structures  that can be regarded as multi-input multi-output (MIMO) black boxes. As such, interpreting them becomes very difficult. With their heavy deployment in a wide variety of safety-critical domains such as healthcare and autonomous navigation, it is becoming increasingly necessary to build trust and accountability in their use [@zhang2021survey].

One approach is to leverage surrogate models such as decision trees [@yang2018deepneuraldecisiontrees] and Gaussian processes [@pmlr-v216-li23c] to increase interpretability, or use sophisticated model-agnostic methods such as LIME [@ribeiro2016should] or SHAP [@lundberg2017unifiedapproachinterpretingmodel]. Another promising approach is neural network verification, which generates mathematical guarantees that a neural network respects its safety specifications, such as input-output bounds.

With the advent of friendly competitions such as International Verification of Neural Networks Competition (VNN-COMP) [@brix2023fourthinternationalverificationneural], the problem of safety verification of neural networks is becoming more standardized, and we are seeing a shift from theoretical approaches to practical, measurable efforts. This tool, much like current state-of-the-art such as Marabou [@wu2024marabou], $\alpha,\beta$-crown [@abcrown] and NeuralSAT [@duong2023dpll], is an attempt in this direction.

# Methodology

# Acknowledgements

The author acknowledges Shubhajit Roy, Senior Research Fellow, IIT Gandhinagar and Avishek Lahiri, Senior Research Fellow, IACS Kolkata for their valuable input.
The author also acknowledges Dr. Rajarshi Ray, Associate Professor, IACS Kolkata for his support and feedback.

# References
