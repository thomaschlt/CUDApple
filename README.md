# CUDApple

Welcome to CUDApple, a project aimed at translating CUDA kernels into Metal shaders for execution on Apple Silicon devices. This project is a work in progress, with some components fully functional and others still under development.

![CUDApple presentation](src/docs/image.png)

## What's Working

- **Matrix and Vector Addition**: These operations have been successfully translated and executed using Metal.
- **Softmax Kernel**: Currently a work in progress. The translation is partially complete, but execution issues are being resolved.

## Getting Started

To try out the project, clone the repository and follow these steps:

1. **Compile the Project**: Use `cargo build` to compile the Rust code.
2. **Run a Kernel**: Use `cd src` and `cargo run -- -i examples/vector_add.cu -d output/ --run -v` to translate and execute a CUDA kernel.

## Core Components

### Abstract Syntax Tree (AST) Structure
The AST is a crucial part of the project, representing the structure of CUDA code in a way that can be manipulated and translated into Metal. You can find the AST implementation in the `parser/unified_ast.rs` file.

### Parser
The parser reads CUDA source files and converts them into an AST. This is the first step in the translation process. The parser is located in the `parser/cuda_grammar.rs` file.

### Generator
The generator takes the AST and produces Metal shader code. This involves mapping CUDA constructs to their Metal equivalents. The generator code is found in the `metal/mod.rs` and `metal/host.rs` files.

## In Progress

- **Multidimensional Support**: I'm working on supporting multidimensional operations, which is crucial for more complex computations.
- **Softmax Emulation**: Emulating a softmax CUDA implementation is underway, aiming to achieve efficient execution on Mac.
- **Complete CNN Implementation**: The goal is to implement an entire CNN in CUDA and run it on Mac. This includes:
  - **Conv2D**
  - **MaxPool2D**
  - **Tanh**
  - **...**

---
Thank you for checking out CUDApple! Don't hesitate to reach out on [X](https://twitter.com/eredictus) if you have any questions or suggestions :)
