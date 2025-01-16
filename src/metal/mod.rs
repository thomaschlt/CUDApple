use crate::parser::unified_ast::{
    Block, CudaProgram, Expression, KernelFunction, Operator, Parameter, Statement, Type,
};
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
        // Add Metal shader header with proper indentation
        writeln!(self.source, "            #include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            #include <metal_math>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            using namespace metal;").map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

        // Process each kernel in the AST with consistent indentation
        for kernel in &program.device_code {
            self.generate_kernel_with_indent(kernel, "            ")?;
        }

        Ok(())
    }

    fn generate_kernel_with_indent(
        &mut self,
        kernel: &KernelFunction,
        indent: &str,
    ) -> Result<(), String> {
        // Write kernel signature with indentation
        writeln!(self.source, "{}kernel void {}(", indent, kernel.name)
            .map_err(|e| e.to_string())?;

        // Generate parameters with proper indentation
        let param_indent = format!("{}    ", indent);
        let mut first = true;
        for (index, param) in kernel.parameters.iter().enumerate() {
            if !first {
                writeln!(self.source, ",").map_err(|e| e.to_string())?;
            }
            first = false;

            write!(self.source, "{}", param_indent).map_err(|e| e.to_string())?;
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
            writeln!(self.source, ",").map_err(|e| e.to_string())?;
        }
        write!(
            self.source,
            "{}uint index [[thread_position_in_grid]]",
            param_indent
        )
        .map_err(|e| e.to_string())?;
        writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

        // Translate kernel body statements with proper indentation
        for stmt in &kernel.body.statements {
            self.translate_statement_with_indent(stmt, &format!("{}    ", indent))?;
        }

        // Close kernel function
        writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

        Ok(())
    }

    fn translate_statement_with_indent(
        &mut self,
        stmt: &Statement,
        indent: &str,
    ) -> Result<(), String> {
        match stmt {
            Statement::VariableDecl(decl) => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                match decl.var_type {
                    Type::Int => write!(self.source, "uint").map_err(|e| e.to_string())?,
                    _ => write!(self.source, "{}", decl.var_type).map_err(|e| e.to_string())?,
                }
                write!(self.source, " {} = ", decl.name).map_err(|e| e.to_string())?;
                if let Some(init) = &decl.initializer {
                    self.translate_expression(init)?;
                }
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            Statement::IfStmt { condition, body } => {
                write!(self.source, "{}if (", indent).map_err(|e| e.to_string())?;
                self.translate_expression(condition)?;
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

                // Translate if body with additional indentation
                for stmt in &body.statements {
                    self.translate_statement_with_indent(stmt, &format!("{}    ", indent))?;
                }

                writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
            }
            Statement::Assign(assign) => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                self.translate_expression(&assign.target)?;
                write!(self.source, " = ").map_err(|e| e.to_string())?;
                self.translate_expression(&assign.value)?;
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unsupported statement: {:?}", stmt)),
        }
        Ok(())
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
            Expression::Variable(name) => {
                write!(self.source, "{}", name).map_err(|e| e.to_string())?;
            }
            Expression::IntegerLiteral(value) => {
                write!(self.source, "{}", value).map_err(|e| e.to_string())?;
            }
            Expression::ThreadIdx(_) | Expression::BlockIdx(_) | Expression::BlockDim(_) => {
                write!(self.source, "index").map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unsupported expression: {:?}", expr)),
        }
        Ok(())
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

    pub fn source(&self) -> &str {
        &self.source
    }
}
