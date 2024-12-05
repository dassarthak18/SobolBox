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

BoxRL-NNV is a tool written in Python for the verification of safety specifications for neural networks. The software takes as inputs
a neural network given in an ONNX (Open Neural Network Exchange) format, and a safety specification given as a VNN-LIB file. ONNX is
an industry standard format for interchange of neural networks between different frameworks such as PyTorch and Tensorflow.
VNN-LIB, likewise, is an international benchmarks standard for the verification of neural networks, which specifies safety properties
as a propositional logic satisfiability problem, in the vein of the SMT-LIB2 format. Thereafter, BoxRL-NNV verifies whether the
given neural network satisfies the safety properties specified by the VNN-LIB file.

# Statement of need

# Methodology

# Acknowledgements

The author acknowledges Shubhajit Roy, Senior Research Fellow, IIT Gandhinagar and Avishek Lahiri, Senior Research Fellow, IACS Kolkata for their valuable input.
The author also acknowledges Dr. Rajarshi Ray, Associate Professor, IACS Kolkata for his support and feedback.

# References
