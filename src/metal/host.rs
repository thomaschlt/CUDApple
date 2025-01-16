pub struct MetalKernelConfig {
    pub grid_size: (u32, u32, u32),
    pub threadgroup_size: (u32, u32, u32),
}

pub struct MetalHostGenerator {
    config: MetalKernelConfig,
    shader: String,
}

impl MetalHostGenerator {
    pub fn new(config: MetalKernelConfig, shader: String) -> Self {
        Self { config, shader }
    }

    pub fn generate_swift_code(&self) -> (String, String) {
        let runner_template = include_str!("./templates/metal_runner.swift");
        let main_template = include_str!("./templates/main.swift");

        let kernel_name = self
            .extract_kernel_name(&self.shader)
            .unwrap_or("vectorAdd".to_string());

        let runner_code = runner_template
            .replace("{{METAL_SHADER}}", &self.shader)
            .replace("{{KERNEL_NAME}}", &kernel_name)
            .replace("{{GRID_SIZE}}", &format!("{:?}", self.config.grid_size))
            .replace(
                "{{THREADGROUP_SIZE}}",
                &format!("{:?}", self.config.threadgroup_size),
            );

        (runner_code, main_template.to_string())
    }

    fn extract_kernel_name(&self, shader: &str) -> Option<String> {
        // Simple parser to extract kernel name from Metal shader
        if let Some(line) = shader.lines().find(|l| l.contains("kernel void")) {
            if let Some(name) = line.split("void").nth(1)?.split("(").next() {
                return Some(name.trim().to_string());
            }
        }
        None
    }
}
