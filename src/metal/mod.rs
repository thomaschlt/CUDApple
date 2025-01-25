use host::MetalKernelConfig;

use crate::parser::unified_ast::{
    CudaProgram, Dimension, Expression, KernelFunction, Operator, Statement, Type,
};
use std::fmt::Write;
pub mod host;

#[derive(Debug)]
pub struct MetalShader {
    source: String,
    config: MetalKernelConfig,
    thread_vars_declared: bool,
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            config: MetalKernelConfig {
                dimensions: 1, // Default to 1D
                grid_size: (4096, 1, 1),
                threadgroup_size: (256, 1, 1),
            },
            thread_vars_declared: false,
        }
    }

    pub fn set_config(&mut self, config: MetalKernelConfig) {
        self.config = config;
    }

    pub fn generate(&mut self, program: &CudaProgram) -> Result<(), String> {
        // Generate Metal shader header with proper indentation
        writeln!(self.source, "            #include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            #include <metal_math>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            using namespace metal;\n").map_err(|e| e.to_string())?;

        // Generate each kernel function
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
                    write!(
                        self.source,
                        "constant uint& {} [[buffer({})]]",
                        param.name, index
                    )
                    .map_err(|e| e.to_string())?;
                }
                _ => {
                    return Err(format!(
                        "Unsupported parameter type: {:?}",
                        param.param_type
                    ))
                }
            }
        }

        // Add thread position parameters
        if !first {
            writeln!(self.source, ",").map_err(|e| e.to_string())?;
        }
        match self.config.dimensions {
            1 => {
                write!(
                    self.source,
                    "{}uint32_t index [[thread_position_in_grid]]",
                    param_indent
                )
            }
            2 => {
                write!(
                    self.source,
                    "{}uint2 thread_position_in_grid [[thread_position_in_grid]]",
                    param_indent
                )
            }
            _ => return Err("Unsupported dimensions".to_string()),
        }
        .map_err(|e| e.to_string())?;

        // // Add thread group size parameter if 2D
        // if self.config.dimensions == 2 {
        //     writeln!(self.source, ",").map_err(|e| e.to_string())?;
        //     write!(
        //         self.source,
        //         "{}uint32_t threadgroup_size [[threads_per_threadgroup]]",
        //         param_indent
        //     )
        //     .map_err(|e| e.to_string())?;
        // }

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
                    Type::Int => write!(self.source, "int32_t").map_err(|e| e.to_string())?,
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
            Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                // Write the for loop header
                write!(self.source, "{}for (", indent).map_err(|e| e.to_string())?;

                // Translate initialization
                match &**init {
                    Statement::VariableDecl(decl) => {
                        let type_str = self.translate_type(&decl.var_type);
                        write!(self.source, "{} {} = ", type_str, decl.name)
                            .map_err(|e| e.to_string())?;
                        if let Some(init_val) = &decl.initializer {
                            self.translate_expression(init_val)?;
                        }
                    }
                    _ => return Err("Unsupported for loop initialization".to_string()),
                }

                // Translate condition and increment
                write!(self.source, "; ").map_err(|e| e.to_string())?;
                self.translate_expression(&condition)?;
                write!(self.source, "; ").map_err(|e| e.to_string())?;

                // Handle increment
                match &**increment {
                    Statement::Assign(assign) => {
                        self.translate_expression(&assign.target)?;
                        write!(self.source, " = ").map_err(|e| e.to_string())?;
                        self.translate_expression(&assign.value)?;
                    }
                    Statement::CompoundAssign {
                        target,
                        operator,
                        value,
                    } => {
                        self.translate_expression(target)?;
                        match operator {
                            Operator::Add => write!(self.source, " += "),
                            Operator::Subtract => write!(self.source, " -= "),
                            Operator::Multiply => write!(self.source, " *= "),
                            Operator::Divide => write!(self.source, " /= "),
                            _ => return Err("Unsupported compound operator".to_string()),
                        }
                        .map_err(|e| e.to_string())?;
                        self.translate_expression(value)?;
                        writeln!(self.source, ";").map_err(|e| e.to_string())?;
                    }
                    _ => return Err("Unsupported for loop increment".to_string()),
                }

                // Close for loop header and translate body
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

                // Translate body with additional indentation
                for stmt in &body.statements {
                    self.translate_statement_with_indent(stmt, &format!("{}    ", indent))?;
                }

                writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
            }
            Statement::CompoundAssign {
                target,
                operator,
                value,
            } => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                self.translate_expression(target)?;
                match operator {
                    Operator::Add => write!(self.source, " += "),
                    Operator::Subtract => write!(self.source, " -= "),
                    Operator::Multiply => write!(self.source, " *= "),
                    Operator::Divide => write!(self.source, " /= "),
                    _ => return Err("Unsupported compound operator".to_string()),
                }
                .map_err(|e| e.to_string())?;
                self.translate_expression(value)?;
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unsupported statement: {:?}", stmt)),
        }
        Ok(())
    }

    fn translate_expression(&mut self, expr: &Expression) -> Result<(), String> {
        match expr {
            Expression::ThreadIdx(dim) | Expression::BlockIdx(dim) => {
                match dim {
                    Dimension::X => write!(self.source, "col"),
                    Dimension::Y => write!(self.source, "row"),
                    _ => return Err("Unsupported thread index dimension".to_string()),
                }
                .map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::ArrayAccess { array, index } => {
                self.translate_expression(array)?;
                write!(self.source, "[").map_err(|e| e.to_string())?;
                self.translate_expression(index)?;
                write!(self.source, "]").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::BinaryOp(lhs, op, rhs) => {
                if let (
                    Expression::BinaryOp(inner_lhs, Operator::Multiply, inner_rhs),
                    Operator::Add,
                    _,
                ) = (&**lhs, op, &**rhs)
                {
                    if is_thread_index_component(inner_lhs)
                        && is_thread_index_component(inner_rhs)
                        && is_thread_index_component(rhs)
                    {
                        match self.config.dimensions {
                            1 => {
                                write!(self.source, "index").map_err(|e| e.to_string())?;
                            }
                            2 => {
                                if is_x_component(rhs) {
                                    write!(self.source, "thread_position_in_grid.x")
                                        .map_err(|e| e.to_string())?;
                                } else if is_y_component(rhs) {
                                    write!(self.source, "thread_position_in_grid.y")
                                        .map_err(|e| e.to_string())?;
                                }
                            }
                            _ => return Err("Unsupported dimensions".to_string()),
                        }
                        return Ok(());
                    }
                }
                self.translate_expression(lhs)?;
                write!(self.source, " {} ", Self::operator_to_string(op))
                    .map_err(|e| e.to_string())?;
                self.translate_expression(rhs)?;
                return Ok(());
            }
            Expression::Variable(name) => {
                write!(self.source, "{}", name).map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::IntegerLiteral(value) => {
                write!(self.source, "{}", value).map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::FloatLiteral(value) => {
                if value.fract() == 0.0 {
                    write!(self.source, "{}.0f", value).map_err(|e| e.to_string())?;
                } else {
                    write!(self.source, "{}f", value).map_err(|e| e.to_string())?;
                }
                return Ok(());
            }
            Expression::BlockDim(component) => {
                match component {
                    Dimension::X => write!(self.source, "threads_per_threadgroup.x")
                        .map_err(|e| e.to_string())?,
                    Dimension::Y => write!(self.source, "threads_per_threadgroup.y")
                        .map_err(|e| e.to_string())?,
                    _ => return Err("Only x and y components supported for BlockDim".to_string()),
                }
                return Ok(());
            }
            Expression::MathFunction { name, arguments } => match name.as_str() {
                "max" => {
                    write!(self.source, "max(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0])?;
                    write!(self.source, ", ").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[1])?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "expf" => {
                    write!(self.source, "exp(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0])?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                _ => return Err(format!("Unsupported math function: {:?}", name)),
            },
            Expression::Infinity => {
                write!(self.source, "INFINITY").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::NegativeInfinity => {
                write!(self.source, "-INFINITY").map_err(|e| e.to_string())?;
                return Ok(());
            }
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
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn translate_type(&self, t: &Type) -> String {
        match t {
            Type::Int => "int".to_string(),
            Type::Float => "float".to_string(),
            Type::Void => "void".to_string(),
            Type::Pointer(inner) => self.translate_type(inner),
            _ => "float".to_string(), // Default fallback
        }
    }
}

fn is_thread_index_component(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::ThreadIdx(_) | Expression::BlockIdx(_) | Expression::BlockDim(_)
    )
}

fn is_x_component(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::ThreadIdx(Dimension::X) | Expression::BlockIdx(Dimension::X)
    )
}

fn is_y_component(expr: &Expression) -> bool {
    matches!(
        expr,
        Expression::ThreadIdx(Dimension::Y) | Expression::BlockIdx(Dimension::Y)
    )
}
