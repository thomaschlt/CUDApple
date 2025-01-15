use crate::parser::unified_ast::{
    Block, CudaProgram, Expression, KernelFunction, Operator, Parameter, Qualifier, Statement, Type,
};
use std::fmt::Write;
pub mod host;
#[cfg(test)]
mod tests;

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
        writeln!(self.source, "#include <metal_math>").map_err(|e| e.to_string())?;
        writeln!(self.source, "using namespace metal;").map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

        // Process each kernel in the AST
        for kernel in &program.device_code {
            self.generate_kernel(kernel)?;
        }

        Ok(())
    }

    fn generate_kernel(&mut self, kernel: &KernelFunction) -> Result<(), String> {
        // Write kernel signature
        writeln!(self.source, "kernel void {}(", kernel.name).map_err(|e| e.to_string())?;

        // Generate parameters
        self.translate_parameters(kernel)?;

        // Begin kernel body
        writeln!(self.source, ") {{").map_err(|e| e.to_string())?;
        writeln!(self.source, "}}").map_err(|e| e.to_string())?;

        Ok(())
    }

    fn translate_parameters(&mut self, kernel: &KernelFunction) -> Result<(), String> {
        let mut first = true;
        for (index, param) in kernel.parameters.iter().enumerate() {
            if !first {
                write!(self.source, ",\n    ").map_err(|e| e.to_string())?;
            }
            first = false;

            // Add buffer attribute for pointer types
            match &param.param_type {
                Type::Pointer(_) => {
                    write!(
                        self.source,
                        "device float* {} [[buffer({})]]",
                        param.name, index
                    )
                    .map_err(|e| e.to_string())?;
                }
                Type::Int => {
                    write!(self.source, "uint {}", param.name).map_err(|e| e.to_string())?;
                }
                _ => {
                    return Err(format!(
                        "Unsupported parameter type: {:?}",
                        param.param_type
                    ))
                }
            }
        }

        // Add thread position parameter
        if !first {
            write!(self.source, ",\n    ").map_err(|e| e.to_string())?;
        }
        write!(self.source, "uint index [[thread_position_in_grid]]").map_err(|e| e.to_string())?;

        Ok(())
    }

    fn translate_thread_index(&mut self, expr: &Expression) -> Result<(), String> {
        // Convert: blockIdx.x * blockDim.x + threadIdx.x
        // To: index
        write!(self.source, "index").map_err(|e| e.to_string())
    }

    fn translate_block(&mut self, block: &Block) -> Result<(), String> {
        for stmt in &block.statements {
            self.translate_statement(stmt)?;
        }
        Ok(())
    }

    fn translate_statement(&mut self, stmt: &Statement) -> Result<(), String> {
        match stmt {
            Statement::IfStmt { condition, body } => {
                write!(self.source, "if (").map_err(|e| e.to_string())?;
                self.translate_expression(condition)?;
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;
                self.translate_block(body)?;
                writeln!(self.source, "}}").map_err(|e| e.to_string())?;
                Ok(())
            }
            Statement::VariableDecl(decl) => {
                write!(self.source, "{} {} = ", decl.var_type, decl.name)
                    .map_err(|e| e.to_string())?;
                if let Some(init) = &decl.initializer {
                    self.translate_expression(init)?;
                }
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
                Ok(())
            }
            Statement::Assign(assign) => {
                self.translate_expression(&assign.target)?;
                write!(self.source, " = ").map_err(|e| e.to_string())?;
                self.translate_expression(&assign.value)?;
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
                Ok(())
            }
            Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                write!(self.source, "for (").map_err(|e| e.to_string())?;
                self.translate_statement(init)?;
                self.translate_expression(condition)?;
                write!(self.source, "; ").map_err(|e| e.to_string())?;
                self.translate_statement(increment)?;
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;
                self.translate_block(body)?;
                writeln!(self.source, "}}").map_err(|e| e.to_string())?;
                Ok(())
            }
            Statement::Include(_)
            | Statement::Empty
            | Statement::CompoundAssign { .. }
            | Statement::MacroCall { .. }
            | Statement::MacroDefinition(_) => Ok(()),
        }
    }

    fn operator_to_string(op: &Operator) -> &'static str {
        match op {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Multiply => "*",
            Operator::Divide => "/",
            Operator::LessThan => "<",
        }
    }

    fn translate_expression(&mut self, expr: &Expression) -> Result<(), String> {
        match expr {
            Expression::ArrayAccess { array, index } => {
                self.translate_expression(array)?;
                write!(self.source, "[").map_err(|e| e.to_string())?;
                self.translate_expression(index)?;
                write!(self.source, "]").map_err(|e| e.to_string())?;
            }
            Expression::BinaryOp(lhs, op, rhs) => {
                self.translate_expression(lhs)?;
                write!(self.source, " {} ", Self::operator_to_string(op))
                    .map_err(|e| e.to_string())?;
                self.translate_expression(rhs)?;
            }
            _ => return Err(format!("Unsupported expression: {:?}", expr)),
        }
        Ok(())
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}
