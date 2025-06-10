use crate::parser::unified_ast::{KernelFunction, Type};

#[derive(Debug, Clone)]
pub struct MetalKernelConfig {
    pub grid_size: (u32, u32, u32),
    pub threadgroup_size: (u32, u32, u32),
    pub dimensions: u32,
}

pub struct MetalHostGenerator {
    config: MetalKernelConfig,
    shader: String,
    kernel: KernelFunction,
}

impl MetalHostGenerator {
    pub fn new(config: MetalKernelConfig, shader: String, kernel: KernelFunction) -> Self {
        Self {
            config,
            shader,
            kernel,
        }
    }

    pub fn generate_swift_code(&self) -> (String, String) {
        let runner_template = include_str!("./templates/metal_runner.swift");
        let main_template = include_str!("./templates/main.swift");

        let kernel_function = self.generate_kernel_function();
        let parameter_init = self.generate_parameter_initialization();

        // Calculate the output buffer index (last pointer parameter)
        let output_buffer_index = self.calculate_output_buffer_index();

        let (width, height) = match self.config.dimensions {
            1 => ("nil".to_string(), "nil".to_string()),
            2 => {
                // Check for both Conv2D style (H,W) and matrix style (M,N) parameters
                let has_h = self
                    .kernel
                    .parameters
                    .iter()
                    .any(|p| p.name == "H" || p.name == "H_out");
                let has_w = self
                    .kernel
                    .parameters
                    .iter()
                    .any(|p| p.name == "W" || p.name == "W_out");
                let has_m = self.kernel.parameters.iter().any(|p| p.name == "M");
                let has_n = self.kernel.parameters.iter().any(|p| p.name == "N");

                if has_h && has_w {
                    ("Int(W)".to_string(), "Int(H)".to_string())
                } else if has_m && has_n {
                    ("Int(M)".to_string(), "Int(N)".to_string())
                } else {
                    ("1000".to_string(), "1000".to_string())
                }
            }
            _ => panic!("Unsupported dimensions"),
        };

        let runner_code = runner_template
            .replace("{{KERNEL_DEFINITIONS}}", &self.shader)
            .replace("{{KERNEL_NAME}}", &self.kernel.name)
            .replace(
                "}\n\nenum MetalError",
                &format!("}}\n\n{}\n\nenum MetalError", kernel_function),
            )
            .replace("{{GRID_SIZE}}", &format!("{:?}", self.config.grid_size))
            .replace(
                "{{THREADGROUP_SIZE}}",
                &format!("{:?}", self.config.threadgroup_size),
            )
            .replace("{{DIMENSIONS}}", &self.config.dimensions.to_string())
            .replace("{{WIDTH}}", &width)
            .replace("{{HEIGHT}}", &height)
            .replace("{{OUTPUT_BUFFER_INDEX}}", &output_buffer_index.to_string());

        let main_code = main_template
            .replace("{{PARAMETER_INIT}}", &parameter_init)
            .replace("{{KERNEL_CALL}}", &self.generate_kernel_call())
            .replace("{{KERNEL_NAME}}", &self.kernel.name);

        (runner_code, main_code)
    }

    fn generate_kernel_function(&self) -> String {
        let params = self
            .kernel
            .parameters
            .iter()
            .map(|p| match &p.param_type {
                Type::Pointer(_) => format!("{}: [{}]", p.name, self.type_to_swift(&p.param_type)),
                Type::Int => format!("{}: {}", p.name, self.type_to_swift(&p.param_type)),
                _ => format!("{}: {}", p.name, self.type_to_swift(&p.param_type)),
            })
            .collect::<Vec<_>>()
            .join(", ");

        let return_type = self
            .kernel
            .parameters
            .iter()
            .filter(|p| matches!(p.param_type, Type::Pointer(_)))
            .last()
            .map(|p| self.type_to_swift(&p.param_type))
            .unwrap_or_else(|| "Float".to_string());

        let inputs = self
            .kernel
            .parameters
            .iter()
            .map(|p| match &p.param_type {
                Type::Pointer(_) => format!(
                    "(data: {}, type: {}.self)",
                    p.name,
                    self.type_to_swift(&p.param_type)
                ),
                Type::Int => format!(
                    "(data: {}, type: {}.self)",
                    p.name,
                    self.type_to_swift(&p.param_type)
                ),
                _ => format!(
                    "(data: {}, type: {}.self)",
                    p.name,
                    self.type_to_swift(&p.param_type)
                ),
            })
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "extension MetalKernelRunner {{
                func {}({}) throws -> [{}] {{
                    let inputs: [(data: Any, type: Any.Type)] = [{}]
                    return try executeKernel(inputs: inputs, outputType: {}.self)
                }}
            }}",
            self.kernel.name, params, return_type, inputs, return_type
        )
    }

    fn type_to_swift(&self, param_type: &Type) -> String {
        match param_type {
            Type::Float => "Float".to_string(),
            Type::Int => "Int32".to_string(),
            Type::Pointer(inner) => self.type_to_swift(inner),
            _ => "Float".to_string(),
        }
    }

    fn generate_parameter_initialization(&self) -> String {
        let mut init_code = String::new();

        let has_h = self
            .kernel
            .parameters
            .iter()
            .any(|p| p.name == "H" || p.name == "H_out");
        let has_w = self
            .kernel
            .parameters
            .iter()
            .any(|p| p.name == "W" || p.name == "W_out");
        let is_conv2d = has_h && has_w;

        let is_backward = self.kernel.name.contains("backward")
            || self
                .kernel
                .parameters
                .iter()
                .any(|p| p.name.starts_with("grad_"));

        // Check if this is a softmax kernel
        let is_softmax = self
            .kernel
            .parameters
            .iter()
            .any(|p| p.name == "batch_size")
            && self
                .kernel
                .parameters
                .iter()
                .any(|p| p.name == "num_classes");

        let is_linear_backward = self.kernel.name.contains("linear_backward")
            && self
                .kernel
                .parameters
                .iter()
                .any(|p| p.name == "in_features" || p.name == "out_features");

        let is_sgd = self.kernel.name.contains("sgd_optimizer");

        // ADD THIS: Check if this is a training loop kernel
        let is_training_loop = self.kernel.name.contains("training_step")
            || self.kernel.name.contains("training_loop")
            || (self
                .kernel
                .parameters
                .iter()
                .any(|p| p.name == "fc1_weights")
                && self
                    .kernel
                    .parameters
                    .iter()
                    .any(|p| p.name == "fc2_weights")
                && self
                    .kernel
                    .parameters
                    .iter()
                    .any(|p| p.name == "target_labels"));

        match self.config.dimensions {
            1 => {
                if self.kernel.name.contains("conv2d_backward_bias") {
                    init_code.push_str("// Conv2D Backward Bias: simple test case\n");
                    init_code.push_str("let H_out: Int32 = 30\n");
                    init_code.push_str("let W_out: Int32 = 30\n");
                    init_code.push_str("let C_out: Int32 = 16\n");
                    init_code.push_str("let outputSize = Int(H_out) * Int(W_out) * Int(C_out)\n");
                    init_code.push_str("let problemSize = Int(C_out)  // For bias gradients\n");
                    init_code.push_str("let inputSize = outputSize\n");
                } else if is_linear_backward {
                    init_code.push_str("// Linear backward pass: simple test case\n");
                    init_code.push_str("let batch_size: Int32 = 2\n");
                    init_code.push_str("let in_features: Int32 = 3\n");
                    init_code.push_str("let out_features: Int32 = 2\n");
                    init_code
                        .push_str("let input_size = Int(batch_size) * Int(in_features)  // 6\n");
                    init_code
                        .push_str("let weight_size = Int(in_features) * Int(out_features)  // 6\n");
                    init_code
                        .push_str("let output_size = Int(batch_size) * Int(out_features)  // 4\n");

                    if self.kernel.name.contains("input") {
                        init_code
                            .push_str("let problemSize = input_size  // For input gradients\n");
                    } else if self.kernel.name.contains("weights") {
                        init_code
                            .push_str("let problemSize = weight_size  // For weight gradients\n");
                    } else if self.kernel.name.contains("bias") {
                        init_code.push_str(
                            "let problemSize = Int(out_features)  // For bias gradients\n",
                        );
                    } else {
                        init_code.push_str("let problemSize = input_size  // Default\n");
                    }
                } else if is_training_loop {
                    init_code.push_str("let batch_size: Int32 = 4  // Small batch for testing\n");
                    init_code.push_str("let input_size: Int32 = 784  // MNIST: 28x28 = 784\n");
                    init_code.push_str("let hidden_size: Int32 = 128  // Hidden layer size\n");
                    init_code.push_str("let output_size: Int32 = 10   // 10 MNIST classes\n");

                    init_code.push_str("// Calculate buffer sizes\n");
                    init_code.push_str(
                        "let fc1_weight_size = Int(input_size) * Int(hidden_size)  // 784 * 128\n",
                    );
                    init_code.push_str("let fc1_bias_size = Int(hidden_size)  // 128\n");
                    init_code.push_str(
                        "let fc2_weight_size = Int(hidden_size) * Int(output_size)  // 128 * 10\n",
                    );
                    init_code.push_str("let fc2_bias_size = Int(output_size)  // 10\n");
                    init_code.push_str("let activation_size = Int(batch_size) * Int(hidden_size)  // Activations\n");
                    init_code.push_str(
                        "let output_batch_size = Int(batch_size) * Int(output_size)  // Outputs\n",
                    );
                    init_code.push_str(
                        "let input_batch_size = Int(batch_size) * Int(input_size)  // Inputs\n",
                    );

                    init_code.push_str(
                        "let problemSize = fc1_weight_size  // Use largest buffer size\n",
                    );
                    init_code.push_str("let inputSize = input_batch_size\n");
                } else if is_sgd {
                    // SGD optimizer initialization (moved from 2D to 1D)
                    init_code.push_str("// SGD optimizer: smaller test case\n");
                    init_code.push_str("let problemSize = 1000  // Smaller for testing\n");
                    init_code.push_str("let inputSize = problemSize\n");
                    init_code.push_str(
                        "let N: Int32 = Int32(problemSize)  // Size parameter for SGD (scalar)\n",
                    );
                } else {
                    init_code.push_str("let problemSize = 1000000\n");
                    init_code.push_str(
                        "let inputSize = problemSize  // For 1D kernels, inputSize = problemSize\n",
                    );

                    // For softmax kernels, calculate batch_size and num_classes
                    if is_softmax {
                        init_code.push_str(
                            "// Softmax parameters: batch_size * num_classes = problemSize\n",
                        );
                        init_code.push_str("let batch_size: Int32 = 1000\n");
                        init_code.push_str("let num_classes: Int32 = 1000\n");
                    }
                }
            }
            2 => {
                if is_linear_backward {
                    // LINEAR BACKWARD 2D: Matrix operations
                    init_code.push_str("// Linear backward pass: matrix operations\n");
                    init_code.push_str("let batch_size: Int32 = 2\n");
                    init_code.push_str("let in_features: Int32 = 3\n");
                    init_code.push_str("let out_features: Int32 = 2\n");
                    init_code.push_str("let input_size = Int(batch_size) * Int(in_features)\n");
                    init_code.push_str("let weight_size = Int(in_features) * Int(out_features)\n");
                    init_code.push_str("let output_size = Int(batch_size) * Int(out_features)\n");
                    init_code.push_str("let problemSize = max(input_size, weight_size)\n");
                } else if is_conv2d {
                    init_code.push_str("// Conv2D Parameters\n");
                    init_code.push_str("let H: Int32 = 32         // Input height\n");
                    init_code.push_str("let W: Int32 = 32         // Input width\n");
                    init_code.push_str("let C: Int32 = 3          // Input channels\n");
                    init_code.push_str("let K_h: Int32 = 3        // Kernel height\n");
                    init_code.push_str("let K_w: Int32 = 3        // Kernel width\n");
                    init_code.push_str("let pad_h: Int32 = 0      // Padding height\n");
                    init_code.push_str("let pad_w: Int32 = 0      // Padding width\n");
                    init_code.push_str("let stride_h: Int32 = 1   // Stride height\n");
                    init_code.push_str("let stride_w: Int32 = 1   // Stride width\n");
                    init_code.push_str("let H_out: Int32 = H - K_h + 1\n");
                    init_code.push_str("let W_out: Int32 = W - K_w + 1\n");

                    // Define C_in and C_out for all Conv2D operations
                    if is_backward && self.kernel.name.contains("conv2d_backward") {
                        init_code.push_str("let C_in: Int32 = C     // Same as input channels\n");
                        init_code
                            .push_str("let C_out: Int32 = 16   // Reasonable output channels\n");

                        // Correct size calculations for backward kernels
                        init_code.push_str("let inputSize = Int(H) * Int(W) * Int(C_in)\n"); // grad_input size
                        init_code
                            .push_str("let outputSize = Int(H_out) * Int(W_out) * Int(C_out)\n"); // grad_output size
                        init_code.push_str(
                            "let weightSize = Int(K_h) * Int(K_w) * Int(C_in) * Int(C_out)\n",
                        ); // weights size
                    } else {
                        // Forward kernels
                        init_code.push_str("let inputSize = Int(H) * Int(W) * Int(C)\n");
                        init_code.push_str("let kernelSize = Int(K_h) * Int(K_w) * Int(C)\n");
                        init_code.push_str("let outputSize = Int(H_out) * Int(W_out) * Int(C)\n");
                    }
                } else if is_training_loop {
                    // TRAINING LOOP: Neural network training infrastructure
                    init_code.push_str("// Training loop: MNIST neural network\n");
                    init_code.push_str("let batch_size: Int32 = 4  // Small batch for testing\n");
                    init_code.push_str("let input_size: Int32 = 784  // MNIST: 28x28 = 784\n");
                    init_code.push_str("let hidden_size: Int32 = 128  // Hidden layer size\n");
                    init_code.push_str("let output_size: Int32 = 10   // 10 MNIST classes\n");

                    init_code.push_str("// Calculate buffer sizes\n");
                    init_code.push_str(
                        "let fc1_weight_size = Int(input_size) * Int(hidden_size)  // 784 * 128\n",
                    );
                    init_code.push_str("let fc1_bias_size = Int(hidden_size)  // 128\n");
                    init_code.push_str(
                        "let fc2_weight_size = Int(hidden_size) * Int(output_size)  // 128 * 10\n",
                    );
                    init_code.push_str("let fc2_bias_size = Int(output_size)  // 10\n");
                    init_code.push_str("let activation_size = Int(batch_size) * Int(hidden_size)  // Activations\n");
                    init_code.push_str(
                        "let output_batch_size = Int(batch_size) * Int(output_size)  // Outputs\n",
                    );
                    init_code.push_str(
                        "let input_batch_size = Int(batch_size) * Int(input_size)  // Inputs\n",
                    );

                    init_code.push_str(
                        "let problemSize = fc1_weight_size  // Use largest buffer size\n",
                    );
                    init_code.push_str("let inputSize = input_batch_size\n");
                } else if is_sgd {
                    // SGD optimizer initialization
                    init_code.push_str("// SGD optimizer: smaller test case\n");
                    init_code.push_str("let problemSize = 1000  // Smaller for testing\n");
                    init_code.push_str("let inputSize = problemSize\n");
                    init_code.push_str(
                        "let N: Int32 = Int32(problemSize)  // Size parameter for SGD (scalar)\n",
                    );
                } else {
                    // Matrix initialization
                    init_code.push_str("let M = 1000\n");
                    init_code.push_str("let N = 1000\n");
                    init_code.push_str("let problemSize = Int(M) * Int(N)\n");
                    init_code.push_str("let inputSize = problemSize  // For matrix kernels, inputSize = problemSize\n");
                }
            }
            _ => panic!("Unsupported dimensions"),
        }

        // Handle buffer initialization
        for param in &self.kernel.parameters {
            let buffer_size = if is_training_loop {
                // TRAINING LOOP: Specific buffer sizes for each parameter
                match param.name.as_str() {
                    "input_batch" => "input_batch_size",
                    "target_labels" => "output_batch_size",
                    "fc1_weights" | "grad_fc1_weights" => "fc1_weight_size",
                    "fc1_bias" | "grad_fc1_bias" => "fc1_bias_size",
                    "fc2_weights" | "grad_fc2_weights" => "fc2_weight_size",
                    "fc2_bias" | "grad_fc2_bias" => "fc2_bias_size",
                    "fc1_output" => "activation_size",
                    "fc2_output" | "predictions" => "output_batch_size",
                    "loss_output" => "Int(batch_size)",
                    "learning_rate" => "1", // Single learning rate value
                    // Integer parameters should be single-element arrays
                    "batch_size" | "input_size" | "hidden_size" | "output_size" => "1",
                    _ => "input_batch_size", // fallback
                }
            } else if is_linear_backward {
                // LINEAR BACKWARD: Specific buffer sizes
                match param.name.as_str() {
                    "input" | "grad_input" => "input_size",
                    "weights" | "grad_weights" => "weight_size",
                    "grad_output" => {
                        if self.kernel.name.contains("linear_backward_bias") {
                            "output_size" // grad_output for bias kernel
                        } else {
                            "output_size" // grad_output for other kernels
                        }
                    }
                    "grad_bias" => "Int(out_features)", // grad_bias is only [out_features]
                    "bias" => "Int(out_features)",
                    _ => "input_size", // fallback
                }
            } else if is_conv2d {
                match param.name.as_str() {
                    "grad_input" | "input" => "inputSize",
                    "grad_output" | "output" => "outputSize",
                    "weights" | "kernel" => {
                        if is_backward {
                            "weightSize"
                        } else {
                            "kernelSize"
                        }
                    }
                    "bias" => "Int(C_out)",
                    _ => "inputSize", // fallback
                }
            } else if is_sgd {
                match param.name.as_str() {
                    "lr" => "1", // Learning rate is single element
                    _ => "problemSize",
                }
            } else {
                "problemSize"
            };

            match &param.param_type {
                Type::Pointer(_) => {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);
                    init_code.push_str(&format!(
                        "var {}: [{}] = Array(repeating: {}(0), count: {})\n",
                        unique_name,
                        self.type_to_swift(&param.param_type),
                        self.type_to_swift(&param.param_type),
                        buffer_size
                    ));
                }
                Type::Int => {
                    // Skip linear backward parameters (already defined above)
                    if is_linear_backward
                        && matches!(
                            param.name.as_str(),
                            "batch_size" | "in_features" | "out_features"
                        )
                    {
                        continue;
                    }
                    // Skip Conv2D parameters as they're already defined above
                    if is_conv2d
                        && matches!(
                            param.name.as_str(),
                            "H" | "W"
                                | "C"
                                | "C_in"
                                | "C_out"
                                | "K_h"
                                | "K_w"
                                | "H_out"
                                | "W_out"
                                | "pad_h"
                                | "pad_w"
                                | "stride_h"
                                | "stride_w"
                        )
                    {
                        continue;
                    }
                    // Skip matrix parameters
                    if param.name == "M" || param.name == "N" {
                        continue;
                    }
                    // Skip softmax parameters (already defined above)
                    if is_softmax && (param.name == "batch_size" || param.name == "num_classes") {
                        continue;
                    }

                    // For SGD: Handle N as scalar, not array
                    if is_sgd && param.name == "N" {
                        continue; // N is already defined as scalar above
                    }

                    let unique_name = format!("{}_{}", self.kernel.name, param.name);

                    init_code.push_str(&format!(
                        "var {}: [{}] = Array(repeating: {}(0), count: {})\n",
                        unique_name,
                        self.type_to_swift(&param.param_type),
                        self.type_to_swift(&param.param_type),
                        buffer_size
                    ));
                }
                Type::Float => {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);
                    init_code.push_str(&format!(
                        "var {}: [{}] = Array(repeating: {}(0), count: {})\n",
                        unique_name,
                        self.type_to_swift(&param.param_type),
                        self.type_to_swift(&param.param_type),
                        buffer_size
                    ))
                }
                _ => {}
            }
        }

        // Initialize array data
        if is_linear_backward && self.kernel.name.contains("linear_backward_bias") {
            // Special handling for bias kernel - separate loops for different array sizes
            init_code.push_str("// Initialize grad_output safely\n");
            let grad_output_unique = format!("{}_grad_output", self.kernel.name);
            let grad_bias_unique = format!("{}_grad_bias", self.kernel.name);

            init_code.push_str(&format!(
                "for i in 0..<min(output_size, {}.count) {{\n",
                grad_output_unique
            ));
            init_code.push_str(&format!(
                "    if i == 0 {{ {}[i] = 1.0 }}  // Batch 0: [1.0, 1.0]\n",
                grad_output_unique
            ));
            init_code.push_str(&format!(
                "    if i == 1 {{ {}[i] = 1.0 }}\n",
                grad_output_unique
            ));
            init_code.push_str(&format!(
                "    if i == 2 {{ {}[i] = 1.0 }}  // Batch 1: [1.0, 1.0]\n",
                grad_output_unique
            ));
            init_code.push_str(&format!(
                "    if i == 3 {{ {}[i] = 1.0 }}\n",
                grad_output_unique
            ));
            init_code.push_str("}\n");
            init_code.push_str("// Initialize grad_bias safely\n");
            init_code.push_str(&format!(
                "for i in 0..<min(Int(out_features), {}.count) {{\n",
                grad_bias_unique
            ));
            init_code.push_str(&format!("    {}[i] = 0.0\n", grad_bias_unique));
            init_code.push_str("}\n");
        } else {
            // Original single loop for all other kernels
            let loop_size = if is_linear_backward {
                "input_size" // For other linear backward kernels
            } else if is_conv2d {
                "inputSize"
            } else {
                "problemSize"
            };
            init_code.push_str(&format!("\nfor i in 0..<{} {{\n", loop_size));

            // Original single loop for other kernels
            for param in &self.kernel.parameters {
                if let Type::Pointer(_) = param.param_type {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);

                    if is_linear_backward {
                        // LINEAR BACKWARD: Special test data
                        match param.name.as_str() {
                            "input" => {
                                init_code
                                    .push_str("    // Input matrix: [batch_size, in_features]\n");
                                init_code.push_str(&format!(
                                    "    if i == 0 {{ {}[i] = 1.0 }}  // Batch 0: [1, 2, 3]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 1 {{ {}[i] = 2.0 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 2 {{ {}[i] = 3.0 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 3 {{ {}[i] = 4.0 }}  // Batch 1: [4, 5, 6]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 4 {{ {}[i] = 5.0 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 5 {{ {}[i] = 6.0 }}\n",
                                    unique_name
                                ));
                            }
                            "weights" => {
                                init_code.push_str(
                                    "    // Weight matrix: [in_features, out_features]\n",
                                );
                                init_code.push_str(&format!(
                                    "    if i == 0 {{ {}[i] = 0.1 }}  // [0.1, 0.2]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 1 {{ {}[i] = 0.2 }}  // [0.3, 0.4]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 2 {{ {}[i] = 0.3 }}  // [0.5, 0.6]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 3 {{ {}[i] = 0.4 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 4 {{ {}[i] = 0.5 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 5 {{ {}[i] = 0.6 }}\n",
                                    unique_name
                                ));
                            }
                            "grad_output" => {
                                init_code.push_str(
                                    "    // Gradient from next layer: [batch_size, out_features]\n",
                                );
                                init_code.push_str(&format!(
                                    "    if i == 0 {{ {}[i] = 1.0 }}  // Batch 0: [1.0, 1.0]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 1 {{ {}[i] = 1.0 }}\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 2 {{ {}[i] = 1.0 }}  // Batch 1: [1.0, 1.0]\n",
                                    unique_name
                                ));
                                init_code.push_str(&format!(
                                    "    if i == 3 {{ {}[i] = 1.0 }}\n",
                                    unique_name
                                ));
                            }
                            "grad_input" | "grad_weights" | "grad_bias" => {
                                init_code.push_str("    // Output gradients (initialized to 0)\n");
                                init_code.push_str(&format!(
                                    "    if i < Int(out_features) {{ {}[i] = 0.0 }}\n",
                                    unique_name
                                ));
                            }
                            _ => {
                                init_code.push_str(&format!("    {}[i] = 0.0\n", unique_name));
                            }
                        }
                    } else if is_sgd {
                        // SGD optimizer: Special test data (moved before is_backward)
                        match param.name.as_str() {
                            "weights" => {
                                init_code.push_str(&format!("    {}[i] = Float.random(in: -0.1...0.1)  // Initial weights\n", unique_name));
                            }
                            "grad_weights" => {
                                init_code.push_str(&format!("    {}[i] = Float.random(in: -0.01...0.01)  // Weight gradients\n", unique_name));
                            }
                            "lr" => {
                                continue;
                            }
                            _ => {
                                init_code.push_str(&format!("    {}[i] = 0.0\n", unique_name));
                            }
                        }
                    } else if is_training_loop {
                        // TRAINING LOOP: Add bounds checking for each array
                        let array_bounds = match param.name.as_str() {
                            "input_batch" => "input_batch_size",
                            "target_labels" => "output_batch_size",
                            "fc1_weights" | "grad_fc1_weights" => "fc1_weight_size",
                            "fc1_bias" | "grad_fc1_bias" => "fc1_bias_size",
                            "fc2_weights" | "grad_fc2_weights" => "fc2_weight_size",
                            "fc2_bias" | "grad_fc2_bias" => "fc2_bias_size",
                            "fc1_output" => "activation_size",
                            "fc2_output" | "predictions" => "output_batch_size",
                            "loss_output" => "Int(batch_size)",
                            "learning_rate" => "1",
                            _ => "input_batch_size",
                        };

                        init_code.push_str(&format!("    if i < {} {{\n", array_bounds));

                        match param.name.as_str() {
                            "input_batch" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = Float.random(in: 0.0...1.0)  // MNIST-like input\n",
                                    unique_name
                                ));
                            }
                            "target_labels" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = (i % 10 == 0) ? 1.0 : 0.0  // One-hot labels\n",
                                    unique_name
                                ));
                            }
                            "fc1_weights" | "fc2_weights" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = Float.random(in: -0.1...0.1)  // Xavier init\n",
                                    unique_name
                                ));
                            }
                            "fc1_bias" | "fc2_bias" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = 0.01  // Small bias values\n",
                                    unique_name
                                ));
                            }
                            "learning_rate" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = 0.001  // Learning rate\n",
                                    unique_name
                                ));
                            }
                            "grad_fc1_weights" | "grad_fc2_weights" | "grad_fc1_bias"
                            | "grad_fc2_bias" => {
                                init_code.push_str(&format!(
                                    "        {}[i] = 0.0  // Zero gradients\n",
                                    unique_name
                                ));
                            }
                            _ => {
                                init_code.push_str(&format!(
                                    "        {}[i] = 0.0  // Zero outputs\n",
                                    unique_name
                                ));
                            }
                        }

                        init_code.push_str("    }\n");
                    } else if is_backward {
                        // Keep the existing backward logic for other kernels
                        // Existing initialization for forward pass kernels
                        if param.name.contains("output") || param.name.contains("result") {
                            init_code.push_str(&format!(
                                "    if i < {} {{ {}[i] = 0.0 }}\n",
                                if is_conv2d {
                                    "outputSize"
                                } else {
                                    "problemSize"
                                },
                                unique_name
                            ));
                        } else if param.name == "bias" {
                            init_code.push_str(&format!(
                                "    if i < 1 {{ {}[i] = 0.1 }}\n",
                                unique_name
                            ));
                        } else {
                            let buffer_limit = if is_conv2d {
                                if param.name == "kernel" || param.name == "weight" {
                                    "kernelSize"
                                } else {
                                    "inputSize"
                                }
                            } else {
                                "inputSize"
                            };
                            init_code.push_str(&format!(
                                "    if i < {} {{ {}[i] = Float.random(in: -1.0...1.0) }}\n",
                                buffer_limit, unique_name
                            ));
                        }
                    }
                }
            }
            init_code.push_str("}\n"); // Close the single loop for other kernels
        }

        // Initialize lr parameter for SGD (outside the loop since it's only 1 element)
        if is_sgd {
            let lr_unique = format!("{}_lr", self.kernel.name);
            init_code.push_str(&format!("{}[0] = 0.01  // Learning rate\n", lr_unique));
        }

        // Initialize integer parameter arrays with their scalar values for training loop
        if is_training_loop {
            init_code.push_str("\n// Initialize integer parameter arrays with scalar values\n");
            for param in &self.kernel.parameters {
                if matches!(param.param_type, Type::Int) {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);
                    match param.name.as_str() {
                        "batch_size" => {
                            init_code.push_str(&format!("{}[0] = batch_size\n", unique_name));
                        }
                        "input_size" => {
                            init_code.push_str(&format!("{}[0] = input_size\n", unique_name));
                        }
                        "hidden_size" => {
                            init_code.push_str(&format!("{}[0] = hidden_size\n", unique_name));
                        }
                        "output_size" => {
                            init_code.push_str(&format!("{}[0] = output_size\n", unique_name));
                        }
                        _ => {}
                    }
                }
            }
        }

        init_code.push_str("\n");

        init_code.push_str("print(\"\\n=== Input Values ===\\n\")\n");
        if is_conv2d {
            // Check what parameters this kernel actually has
            if self.kernel.name.contains("conv2d_backward_bias") {
                // Only has output dimensions for bias kernel
                init_code
                    .push_str("print(\"Output dimensions: \\(H_out) x \\(W_out) x \\(C_out)\")\n");
            } else {
                // Full conv2d kernels have input, kernel, and output dimensions
                init_code.push_str("print(\"Input dimensions: \\(H) x \\(W) x \\(C)\")\n");
                init_code.push_str("print(\"Kernel dimensions: \\(K_h) x \\(K_w)\")\n");
                init_code.push_str("print(\"Output dimensions: \\(H_out) x \\(W_out) x \\(C)\")\n");
            }
        }

        let loop_var = if is_linear_backward {
            "input_size" // Use input_size for linear backward
        } else if is_conv2d {
            "inputSize" // Use inputSize for conv2d
        } else {
            "inputSize" // Use inputSize for others
        };

        init_code.push_str(&format!("for i in 0..<min(5, {}) {{\n", loop_var));

        for param in &self.kernel.parameters {
            if let Type::Pointer(_) = param.param_type {
                let unique_name = format!("{}_{}", self.kernel.name, param.name);
                let max_size = match param.name.as_str() {
                    "bias" => format!("min(1, {}.count)", unique_name),
                    _ => format!("min(5, {}.count)", unique_name),
                };
                init_code.push_str(&format!(
                    "    if i < {} {{ print(\"{}[\\(i)] = \\({}[i])\") }}\n",
                    max_size, param.name, unique_name
                ));
            }
        }
        init_code.push_str("}\n");

        init_code
    }

    fn generate_kernel_call(&self) -> String {
        let mut inputs = Vec::new();
        let is_sgd = self.kernel.name.contains("sgd_optimizer");
        let is_linear_backward = self.kernel.name.contains("linear_backward");
        let is_conv2d = self
            .kernel
            .parameters
            .iter()
            .any(|p| p.name == "H" || p.name == "H_out");
        let is_softmax = self
            .kernel
            .parameters
            .iter()
            .any(|p| p.name == "batch_size")
            && self
                .kernel
                .parameters
                .iter()
                .any(|p| p.name == "num_classes");

        for param in &self.kernel.parameters {
            let unique_name = format!("{}_{}", self.kernel.name, param.name);
            match &param.param_type {
                Type::Pointer(_) => {
                    inputs.push(format!("(data: {}, type: Float.self)", unique_name));
                }
                Type::Int => {
                    // Special cases for scalar integer parameters (not arrays)
                    let use_scalar = (is_sgd && param.name == "N")
                        || param.name == "M"
                        || param.name == "N"
                        || (is_linear_backward
                            && matches!(
                                param.name.as_str(),
                                "batch_size" | "in_features" | "out_features"
                            ))
                        || (is_conv2d
                            && matches!(
                                param.name.as_str(),
                                "H" | "W"
                                    | "C"
                                    | "C_in"
                                    | "C_out"
                                    | "K_h"
                                    | "K_w"
                                    | "H_out"
                                    | "W_out"
                                    | "pad_h"
                                    | "pad_w"
                                    | "stride_h"
                                    | "stride_w"
                            ))
                        || (is_softmax
                            && matches!(param.name.as_str(), "batch_size" | "num_classes"));

                    if use_scalar {
                        // Use scalar variable directly
                        inputs.push(format!("(data: UInt32({}), type: UInt32.self)", param.name));
                    } else {
                        // Use array element
                        inputs.push(format!(
                            "(data: UInt32({}[0]), type: UInt32.self)",
                            unique_name
                        ));
                    }
                }
                _ => {}
            }
        }

        let mut code = String::new();
        code.push_str("// Create inputs array to help Swift compiler with type checking\n");
        code.push_str("let kernelInputs: [(data: Any, type: Any.Type)] = [\n");

        for (i, input) in inputs.iter().enumerate() {
            if i == inputs.len() - 1 {
                code.push_str(&format!("    {}\n", input));
            } else {
                code.push_str(&format!("    {},\n", input));
            }
        }

        code.push_str("]\n");
        code.push_str(
            "let result = try runner.executeKernel(inputs: kernelInputs, outputType: Float.self)",
        );

        code
    }

    fn calculate_output_buffer_index(&self) -> usize {
        if self.kernel.name.contains("sgd_optimizer") {
            return 0;
        }

        self.kernel
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.param_type, Type::Pointer(_)))
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
