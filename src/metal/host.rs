use crate::parser::unified_ast::{KernelFunction, Type};

pub struct MetalKernelConfig {
    pub grid_size: (u32, u32, u32),
    pub threadgroup_size: (u32, u32, u32),
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

        // Generate dynamic kernel function
        let kernel_function = self.generate_kernel_function();

        // Generate dynamic parameter initialization
        let parameter_init = self.generate_parameter_initialization();

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
            );

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

        format!(
            "extension MetalKernelRunner {{
                func {}({}) throws -> [{}] {{
                    let inputs: [(data: Any, type: Any.Type)] = [{}]
                    return try executeKernel(inputs: inputs, outputType: {}.self)
                }}
            }}",
            self.kernel.name,
            params,
            return_type,
            self.kernel
                .parameters
                .iter()
                .filter(|p| matches!(p.param_type, Type::Pointer(_)))
                .map(|p| format!(
                    "(data: {}, type: {}.self)",
                    p.name,
                    self.type_to_swift(&p.param_type)
                ))
                .collect::<Vec<_>>()
                .join(", "),
            return_type
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

        // Add problem size with explicit type
        init_code.push_str("let problemSize = 1000000\n");

        // Initialize arrays for each parameter
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
                    // Create Int32 scalar
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

        // Add initialization loop using problemSize
        init_code.push_str("\nfor i in 0..<problemSize {\n");
        for param in &self.kernel.parameters {
            if let Type::Pointer(_) = param.param_type {
                init_code.push_str(&format!(
                    "    {}[i] = {}(i)\n",
                    param.name,
                    self.type_to_swift(&param.param_type)
                ));
            }
        }
        init_code.push_str("}\n");

        init_code
    }

    fn generate_kernel_call(&self) -> String {
        let params = self
            .kernel
            .parameters
            .iter()
            .map(|p| format!("{}: {}", p.name, p.name))
            .collect::<Vec<_>>()
            .join(", ");

        format!("let result = try runner.{}({})", self.kernel.name, params)
    }
}
