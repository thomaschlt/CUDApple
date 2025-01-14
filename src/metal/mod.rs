use crate::parser::{CudaProgram, Expression, KernelFunction, Qualifier, Statement, Type};
use std::fmt::Write;
pub mod host;

#[derive(Debug)]
pub struct MetalShader {
    source: String,
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
        }
    }

    pub fn generate(&mut self, program: &CudaProgram) -> Result<(), String> {
        // Add Metal shader header
        writeln!(self.source, "#include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(self.source, "using namespace metal;").map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

        // Generate each kernel
        for kernel in &program.device_code {}

        Ok(())
    }

    fn translate_type(&self, t: &Type) -> Result<&'static str, String> {
        match t {
            Type::Int => Ok("int"),
            Type::Float => Ok("float"),
            Type::Pointer(inner) => self.translate_type(inner),
            _ => Err(format!("Unsupported type: {:?}", t)),
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}
