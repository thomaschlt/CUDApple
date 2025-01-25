use crate::parser::unified_ast::{KernelFunction, Type};

#[derive(Debug, Clone)]
pub struct MetalKernelConfig {
    pub grid_size: (u32, u32, u32),
    pub threadgroup_size: (u32, u32, u32),
    pub dimensions: u32, // 1 for 1D, 2 for 2D
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

        // Calculate dimensions and sizes for the kernel config
        let (width, height) = match self.config.dimensions {
            1 => ("nil".to_string(), "nil".to_string()),
            2 => {
                let m = self
                    .kernel
                    .parameters
                    .iter()
                    .find(|p| p.name == "M")
                    .expect("2D kernel requires M parameter");
                let n = self
                    .kernel
                    .parameters
                    .iter()
                    .find(|p| p.name == "N")
                    .expect("2D kernel requires N parameter");
                ("M".to_string(), "N".to_string())
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
            .replace("{{HEIGHT}}", &height);

        let main_code = main_template
            .replace("{{PARAMETER_INIT}}", &parameter_init)
            .replace("{{KERNEL_CALL}}", &self.generate_kernel_call())
            .replace("{{KERNEL_NAME}}", &self.kernel.name);

        (runner_code, main_code)
    }

    fn generate_kernel_function(&self) -> String {
        // Generate parameter list with proper types
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

        // Get return type from last pointer parameter
        let return_type = self
            .kernel
            .parameters
            .iter()
            .filter(|p| matches!(p.param_type, Type::Pointer(_)))
            .last()
            .map(|p| self.type_to_swift(&p.param_type))
            .unwrap_or_else(|| "Float".to_string());

        // Build inputs array including both array and scalar parameters
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
            _ => "Float".to_string(), // Default fallback
        }
    }

    fn generate_parameter_initialization(&self) -> String {
        let mut init_code = String::new();

        // Handle dimensions and problem size
        match self.config.dimensions {
            1 => {
                init_code.push_str("let problemSize = 1000000\n");
            }
            2 => {
                init_code.push_str("let M = 1000\n");
                init_code.push_str("let N = 1000\n");
                init_code.push_str("let problemSize = Int(M) * Int(N)\n"); // Cast to Int for array sizing
            }
            _ => panic!("Unsupported dimensions"),
        }

        // Initialize arrays and scalar parameters
        let mut has_m = false;
        let mut has_n = false;

        // First pass: check if M and N are present
        for param in &self.kernel.parameters {
            if let Type::Int = param.param_type {
                if param.name == "M" {
                    has_m = true;
                } else if param.name == "N" {
                    has_n = true;
                }
            }
        }

        // Initialize M and N if needed for 2D kernels
        if self.config.dimensions == 2 {
            if !has_m {
                init_code.push_str("let M: Int32 = 1000\n");
            }
            if !has_n {
                init_code.push_str("let N: Int32 = 1000\n");
            }
        }

        // Initialize other parameters
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
                        continue; // Skip M and N as they're already handled
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

        // Initialize array values
        init_code.push_str("\nfor i in 0..<problemSize {\n");
        for param in &self.kernel.parameters {
            if let Type::Pointer(_) = param.param_type {
                if param.name.contains("res") || param.name.contains("C") {
                    // Add "C" for matrix output
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

        // Add print statements for first few elements
        init_code.push_str("print(\"\\n=== Input Values ===\\n\")\n");
        init_code.push_str("for i in 0..<5 {\n");
        for param in &self.kernel.parameters {
            if let Type::Pointer(_) = param.param_type {
                init_code.push_str(&format!(
                    "    print(\"{0}[\\(i)] = \\({0}[i])\")\n",
                    param.name
                ));
            }
        }
        init_code.push_str("}\n");

        init_code
    }

    fn generate_kernel_call(&self) -> String {
        let mut inputs = Vec::new();

        // Preserve CUDA kernel parameter order
        for param in &self.kernel.parameters {
            match &param.param_type {
                Type::Pointer(_) => {
                    inputs.push(format!("(data: {}, type: Float.self)", param.name));
                }
                Type::Int => {
                    inputs.push(format!("(data: UInt32({}), type: UInt32.self)", param.name));
                }
                _ => {}
            }
        }

        format!(
            "let result = try runner.executeKernel(inputs: [{}], outputType: Float.self)",
            inputs.join(", ")
        )
    }
}
