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

        let (width, height) = match self.config.dimensions {
            1 => ("nil".to_string(), "nil".to_string()),
            2 => {
                let _m = self
                    .kernel
                    .parameters
                    .iter()
                    .find(|p| p.name == "M")
                    .expect("2D kernel requires M parameter");
                let _n = self
                    .kernel
                    .parameters
                    .iter()
                    .find(|p| p.name == "N")
                    .expect("2D kernel requires N parameter");
                ("M".to_string(), "N".to_string())
            }
            _ => panic!("Unsupported dimensions"),
        };

        // not optimal
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
            .replace("{{HEIGHT}}", &height);

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

        // Check if this is a training loop kernel
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

        if is_training_loop {
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

            // Create arrays with appropriate sizes
            for param in &self.kernel.parameters {
                match &param.param_type {
                    Type::Pointer(_) => {
                        let unique_name = format!("{}_{}", self.kernel.name, param.name);
                        let buffer_size = match param.name.as_str() {
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

                        init_code.push_str(&format!(
                            "var {}: [{}] = Array(repeating: {}(0), count: {})\n",
                            unique_name,
                            self.type_to_swift(&param.param_type),
                            self.type_to_swift(&param.param_type),
                            buffer_size
                        ));
                    }
                    Type::Int => {
                        let unique_name = format!("{}_{}", self.kernel.name, param.name);
                        init_code.push_str(&format!(
                            "var {}: [{}] = Array(repeating: {}(0), count: 1)\n",
                            unique_name,
                            self.type_to_swift(&param.param_type),
                            self.type_to_swift(&param.param_type)
                        ));
                    }
                    _ => {}
                }
            }

            // Initialize integer parameters with scalar values
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

            // Initialize arrays with proper bounds checking
            init_code.push_str("\n// Initialize arrays with bounds checking\n");
            init_code.push_str("let max_size = fc1_weight_size  // Use largest array size for loop\n");
            init_code.push_str("for i in 0..<max_size {\n");

            for param in &self.kernel.parameters {
                if let Type::Pointer(_) = param.param_type {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);
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
                        "grad_fc1_weights" | "grad_fc2_weights" | "grad_fc1_bias" | "grad_fc2_bias" => {
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
                }
            }

            init_code.push_str("}\n\n");
        } else {
            // Original logic for non-training kernels
            match self.config.dimensions {
                1 => {
                    init_code.push_str("let problemSize = 1000000\n");
                }
                2 => {
                    init_code.push_str("let M = 1000\n");
                    init_code.push_str("let N = 1000\n");
                    init_code.push_str("let problemSize = Int(M) * Int(N)\n");
                }
                _ => panic!("Unsupported dimensions"),
            }

            let mut has_m = false;
            let mut has_n = false;

            for param in &self.kernel.parameters {
                if let Type::Int = param.param_type {
                    if param.name == "M" {
                        has_m = true;
                    } else if param.name == "N" {
                        has_n = true;
                    }
                }
            }

            if self.config.dimensions == 2 {
                if !has_m {
                    init_code.push_str("let M: Int32 = 1000\n");
                }
                if !has_n {
                    init_code.push_str("let N: Int32 = 1000\n");
                }
            }

            for param in &self.kernel.parameters {
                match &param.param_type {
                    Type::Pointer(_) => {
                        init_code.push_str(&format!(
                            "var {}: [{}] = Array(repeating: {}(0), count: problemSize)\n",
                            param.name,
                            self.type_to_swift(&param.param_type),
                            self.type_to_swift(&param.param_type)
                        ));
                    }
                    Type::Int => {
                        if param.name == "M" || param.name == "N" {
                            continue;
                        }
                        init_code.push_str(&format!(
                            "let {}: {} = {}(problemSize)\n",
                            param.name,
                            self.type_to_swift(&param.param_type),
                            self.type_to_swift(&param.param_type)
                        ));
                    }
                    _ => {}
                }
            }

            init_code.push_str("for i in 0..<problemSize {\n");
            for param in &self.kernel.parameters {
                if let Type::Pointer(_) = param.param_type {
                    if param.name.contains("res") || param.name.contains("C") {
                        init_code.push_str(&format!("    {}[i] = 0.0\n", param.name));
                    } else {
                        init_code.push_str(&format!(
                            "    {}[i] = Float.random(in: -1.0...1.0)\n",
                            param.name
                        ));
                    }
                }
            }
            init_code.push_str("}\n\n");
        }

        init_code.push_str("print(\"\\n=== Input Values ===\\n\")\n");
        init_code.push_str("for i in 0..<5 {\n");
        for param in &self.kernel.parameters {
            if let Type::Pointer(_) = param.param_type {
                if is_training_loop {
                    let unique_name = format!("{}_{}", self.kernel.name, param.name);
                    init_code.push_str(&format!(
                        "    if i < {}.count {{ print(\"{0}[\\(i)] = \\({1}[i])\") }}\n",
                        param.name, unique_name
                    ));
                } else {
                    init_code.push_str(&format!(
                        "    print(\"{0}[\\(i)] = \\({0}[i])\")\n",
                        param.name
                    ));
                }
            }
        }
        init_code.push_str("}\n");

        init_code
    }

    fn generate_kernel_call(&self) -> String {
        let mut inputs = Vec::new();

        // Check if this is a training loop kernel
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

        for param in &self.kernel.parameters {
            match &param.param_type {
                Type::Pointer(_) => {
                    let array_name = if is_training_loop {
                        format!("{}_{}", self.kernel.name, param.name)
                    } else {
                        param.name.clone()
                    };
                    inputs.push(format!("(data: {}, type: Float.self)", array_name));
                }
                Type::Int => {
                    let array_name = if is_training_loop {
                        format!("{}_{}", self.kernel.name, param.name)
                    } else {
                        param.name.clone()
                    };
                    inputs.push(format!(
                        "(data: UInt32({}[0]), type: UInt32.self)",
                        array_name
                    ));
                }
                _ => {}
            }
        }

        // Break up the long expression to avoid Swift compiler timeout
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
}
