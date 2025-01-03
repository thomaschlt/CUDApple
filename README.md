# CUDAPPLE

An experimental CUDA to Metal source-to-source translator. This project explores the feasibility of automatically converting CUDA kernels to Metal compute shaders.

## Project Overview

The translator works in three main phases:

1. **Parsing**: Converts CUDA source code into an Abstract Syntax Tree (AST)
2. **Analysis**: Analyzes CUDA-specific constructs and their Metal equivalents
3. **Code Generation**: Generates Metal shader code from the AST

## AST Structure

The AST represents CUDA kernels with the following key components:

- Programs containing multiple kernel functions
- Kernel functions with parameters and body blocks
- Variable declarations and assignments
- Thread indexing expressions (threadIdx, blockIdx, blockDim)
- Array access and arithmetic operations
- Control flow statements (if conditions)

## Parser Implementation

The parser uses PEG (Parsing Expression Grammar) to handle CUDA syntax, including:

- Kernel declarations
- Thread indexing (threadIdx, blockIdx)
- Memory operations
- Control flow statements

## Status

Currently in exploration phase, focusing on:

- AST representation of CUDA constructs
- Parsing complex CUDA expressions
- Thread/block index mapping strategy

This is a research project to understand the challenges in CUDA to Metal translation! :)
