use anyhow::Result;

use std::fmt::Write;

use crate::parser::Qualifier;

pub mod host;
#[cfg(test)]
mod tests;

// Metal shader types
#[derive(Debug)]
pub enum MetalType {
    Int,
    Float,
    Buffer(Box<MetalType>),
}

impl MetalType {
    pub fn to_metal_string(&self) -> String {
        match self {
            MetalType::Int => "int".to_string(),
            MetalType::Float => "float".to_string(),
            MetalType::Buffer(inner) => format!("device {}*", inner.to_metal_string()),
        }
    }
}

pub struct MetalShader {
    source: String,
    uses_local_thread_index: bool,
    uses_global_thread_index: bool,
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            uses_local_thread_index: false,
            uses_global_thread_index: false,
        }
    }

    pub fn generate(&mut self, program: &crate::parser::CudaProgram) -> Result<()> {
        // Add Metal shader header with math functions
        writeln!(self.source, "#include <metal_stdlib>")?;
        writeln!(self.source, "#include <metal_math>")?;
        writeln!(self.source, "#include <metal_geometric>")?;
        writeln!(self.source, "using namespace metal;")?;
        writeln!(self.source)?;

        // Add common math constants
        writeln!(self.source, "constant float INFINITY = INFINITY;")?;
        writeln!(self.source)?;

        for kernel in &program.device_code {
            self.generate_kernel(kernel)?;
        }

        Ok(())
    }

    pub fn generate_kernel(&mut self, kernel: &crate::parser::KernelFunction) -> Result<()> {
        // Analyze thread usage in kernel body
        for stmt in &kernel.body.statements {
            self.analyze_statements(stmt);
        }

        write!(self.source, "kernel void {}(\n", kernel.name)?;

        // Generate parameters
        for (i, param) in kernel.parameters.iter().enumerate() {
            self.generate_parameter(param, false)?;
        }

        // Add thread position parameters based on usage
        if self.uses_local_thread_index {
            writeln!(
                self.source,
                "    uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],"
            )?;
        }
        if self.uses_global_thread_index {
            writeln!(
                self.source,
                "    uint thread_position_in_grid [[thread_position_in_grid]]"
            )?;
        }

        writeln!(self.source, ") {{")?;

        self.generate_block(&kernel.body)?;
        writeln!(self.source, "}}\n")?;
        Ok(())
    }

    pub fn generate_parameter(
        &mut self,
        param: &crate::parser::Parameter,
        is_last: bool,
    ) -> Result<()> {
        let metal_type = convert_type(&param.param_type);
        let restrict_qualifier = match param.qualifier {
            Qualifier::Restrict => "device const ",
            Qualifier::None => "",
        };
        write!(
            self.source,
            "    {}{} {} [[buffer({})]]{}\n",
            restrict_qualifier,
            metal_type.to_metal_string(),
            param.name,
            param.name.chars().next().unwrap() as u8 - 'a' as u8,
            if !is_last { "," } else { "," }
        )?;
        Ok(())
    }

    pub fn generate_block(&mut self, block: &crate::parser::Block) -> Result<()> {
        for stmt in &block.statements {
            self.generate_statement(stmt)?;
        }
        Ok(())
    }

    pub fn generate_statement(&mut self, stmt: &crate::parser::Statement) -> Result<()> {
        match stmt {
            crate::parser::Statement::Empty => writeln!(self.source, "    ;")?,
            crate::parser::Statement::VariableDecl(decl) => {
                self.generate_declaration(decl)?;
            }
            crate::parser::Statement::Assign(assign) => {
                write!(self.source, "    ")?;
                self.generate_assignment(assign)?;
                writeln!(self.source, ";")?;
            }
            crate::parser::Statement::IfStmt { condition, body } => {
                write!(self.source, "    if (")?;
                self.generate_expression(condition)?;
                writeln!(self.source, ") {{")?;
                self.generate_block(body)?;
                writeln!(self.source, "    }}")?;
            }
            crate::parser::Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                write!(self.source, "    for (")?;
                self.generate_statement(init)?;
                self.generate_expression(condition)?;
                write!(self.source, "; ")?;
                self.generate_statement(increment)?;
                writeln!(self.source, ") {{")?;
                self.generate_block(body)?;
                writeln!(self.source, "    }}")?;
            }
        }
        Ok(())
    }

    pub fn generate_declaration(&mut self, decl: &crate::parser::Declaration) -> Result<()> {
        let metal_type = convert_type(&decl.var_type);
        write!(
            self.source,
            "    {} {}",
            metal_type.to_metal_string(),
            decl.name
        )?;
        if let Some(init) = &decl.initializer {
            write!(self.source, " = ")?;
            self.generate_expression(init)?;
        }
        writeln!(self.source, ";")?;
        Ok(())
    }

    pub fn generate_assignment(&mut self, assign: &crate::parser::Assignment) -> Result<()> {
        self.generate_expression(&assign.target)?;
        write!(self.source, " = ")?;
        self.generate_expression(&assign.value)?;
        Ok(())
    }

    pub fn generate_expression(&mut self, expr: &crate::parser::Expression) -> Result<()> {
        match expr {
            crate::parser::Expression::Number(n) => write!(self.source, "{}", n)?,
            crate::parser::Expression::IntegerLiteral(i) => write!(self.source, "{}", i)?,
            crate::parser::Expression::Variable(name) => {
                write!(self.source, "{}", name)?;
            }
            crate::parser::Expression::BinaryOp(left, op, right) => {
                self.generate_expression(left)?;
                write!(
                    self.source,
                    " {} ",
                    match op {
                        crate::parser::Operator::Add => "+",
                        crate::parser::Operator::Subtract => "-",
                        crate::parser::Operator::Multiply => "*",
                        crate::parser::Operator::Divide => "/",
                        crate::parser::Operator::LessThan => "<",
                    }
                )?;
                self.generate_expression(right)?;
            }
            crate::parser::Expression::FunctionCall(name, args) => {
                match name.as_str() {
                    "expf" => write!(self.source, "metal::exp")?,
                    "max" => write!(self.source, "metal::fmax")?,
                    _ => write!(self.source, "{}", name)?,
                }
                write!(self.source, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(self.source, ", ")?;
                    }
                    self.generate_expression(arg)?;
                }
                write!(self.source, ")")?;
            }
            crate::parser::Expression::ThreadIdx(_) => {
                write!(self.source, "thread_position_in_threadgroup")?;
            }
            crate::parser::Expression::BlockIdx(_) => {
                write!(self.source, "threadgroup_position_in_grid")?;
            }
            crate::parser::Expression::BlockDim(_) => {
                write!(self.source, "threads_per_threadgroup")?;
            }
            crate::parser::Expression::ArrayAccess { array, index } => {
                self.generate_expression(array)?;
                write!(self.source, "[")?;
                self.generate_expression(index)?;
                write!(self.source, "]")?;
            }
            crate::parser::Expression::SizeOf(t) => {
                write!(self.source, "sizeof({})", t)?;
            }
        }
        Ok(())
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn analyze_thread_usage(&mut self, expr: &crate::parser::Expression) {
        match expr {
            crate::parser::Expression::ThreadIdx(_) => {
                self.uses_local_thread_index = true;
            }
            crate::parser::Expression::BlockIdx(_) | crate::parser::Expression::BlockDim(_) => {
                self.uses_global_thread_index = true;
            }
            _ => {}
        }
    }

    fn analyze_statements(&mut self, stmt: &crate::parser::Statement) {
        match stmt {
            crate::parser::Statement::Empty => {}
            crate::parser::Statement::VariableDecl(decl) => {
                if let Some(init) = &decl.initializer {
                    self.analyze_expression(init);
                }
            }
            crate::parser::Statement::Assign(assign) => {
                self.analyze_expression(&assign.target);
                self.analyze_expression(&assign.value);
            }
            crate::parser::Statement::IfStmt { condition, body } => {
                self.analyze_expression(condition);
                for stmt in &body.statements {
                    self.analyze_statements(stmt);
                }
            }
            crate::parser::Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                self.analyze_statements(init);
                self.analyze_expression(condition);
                self.analyze_statements(increment);
                for stmt in &body.statements {
                    self.analyze_statements(stmt);
                }
            }
        }
    }

    fn analyze_expression(&mut self, expr: &crate::parser::Expression) {
        match expr {
            crate::parser::Expression::Number(_) => {}
            // Direct thread index usage
            crate::parser::Expression::ThreadIdx(_) => {
                self.uses_local_thread_index = true;
            }
            // Block index or dimension implies grid calculation
            crate::parser::Expression::BlockIdx(_) | crate::parser::Expression::BlockDim(_) => {
                self.uses_global_thread_index = true;
            }
            // Recursively analyze binary operations
            crate::parser::Expression::BinaryOp(left, _op, right) => {
                self.analyze_expression(left);
                self.analyze_expression(right);
            }
            // Recursively analyze array access
            crate::parser::Expression::ArrayAccess { array, index } => {
                self.analyze_expression(array);
                self.analyze_expression(index);
            }
            // Other expressions don't affect thread indexing
            _ => {}
        }
    }

    pub fn generate_host_code(&self, config: host::MetalKernelConfig) -> String {
        let host_gen = host::MetalHostGenerator::new(config, self.source().to_string());
        host_gen.generate_swift_code()
    }
}
pub fn convert_type(cuda_type: &crate::parser::Type) -> MetalType {
    match cuda_type {
        crate::parser::Type::Int => MetalType::Int,
        crate::parser::Type::Float => MetalType::Float,
        crate::parser::Type::Pointer(inner) => MetalType::Buffer(Box::new(convert_type(inner))),
        crate::parser::Type::Void => MetalType::Int, // Void not used in our first version
        crate::parser::Type::SizeT => MetalType::Int, // size_t maps to int in Metal
        crate::parser::Type::Dim3 => MetalType::Int, // dim3 maps to int in Metal
    }
}
