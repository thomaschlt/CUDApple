use anyhow::Result;
use std::fmt::Write;

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
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
        }
    }

    pub fn generate(&mut self, program: &crate::parser::CudaProgram) -> Result<()> {
        // Add Metal shader header
        writeln!(self.source, "#include <metal_stdlib>")?;
        writeln!(self.source, "using namespace metal;")?;
        writeln!(self.source)?;

        for kernel in &program.kernels {
            self.generate_kernel(kernel)?;
        }

        Ok(())
    }

    pub fn generate_kernel(&mut self, kernel: &crate::parser::KernelFunction) -> Result<()> {
        // Generate kernel signature
        write!(self.source, "kernel void {}(\n", kernel.name)?;

        // Generate parameters
        for (i, param) in kernel.parameters.iter().enumerate() {
            self.generate_parameter(param, i == kernel.parameters.len() - 1)?;
        }

        // Add thread position parameter
        writeln!(
            self.source,
            "    uint thread_position_in_grid [[thread_position_in_grid]]"
        )?;
        writeln!(self.source, ") {{")?;

        // Generate kernel body
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
        write!(
            self.source,
            "    {} {} [[buffer({})]]{}\n",
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
            crate::parser::Statement::Declaration(decl) => {
                self.generate_declaration(decl)?;
            }
            crate::parser::Statement::Assignment(assign) => {
                write!(self.source, "    ")?;
                self.generate_assignment(assign)?;
                writeln!(self.source, ";")?;
            }
            crate::parser::Statement::If(cond, block) => {
                write!(self.source, "    if (")?;
                self.generate_expression(cond)?;
                writeln!(self.source, ") {{")?;
                self.generate_block(block)?;
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
            crate::parser::Expression::Literal(lit) => match lit {
                crate::parser::Literal::Integer(i) => write!(self.source, "{}", i)?,
            },
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
                        crate::parser::Operator::Mul => "*",
                        crate::parser::Operator::LessThan => "<",
                    }
                )?;
                self.generate_expression(right)?;
            }
            crate::parser::Expression::ThreadIdx(_) => {
                write!(self.source, "thread_position_in_grid")?;
            }
            crate::parser::Expression::BlockIdx(_) => {
                write!(self.source, "(thread_position_in_grid / {})", 256)?; // Assuming fixed block size
            }
            crate::parser::Expression::BlockDim(_) => {
                write!(self.source, "{}", 256)?; // Fixed block size for first version
            }
            crate::parser::Expression::ArrayAccess { array, index } => {
                self.generate_expression(array)?;
                write!(self.source, "[")?;
                self.generate_expression(index)?;
                write!(self.source, "]")?;
            }
        }
        Ok(())
    }

    pub fn source(&self) -> &str {
        &self.source
    }
}
pub fn convert_type(cuda_type: &crate::parser::Type) -> MetalType {
    match cuda_type {
        crate::parser::Type::Int => MetalType::Int,
        crate::parser::Type::Float => MetalType::Float,
        crate::parser::Type::Pointer(inner) => MetalType::Buffer(Box::new(convert_type(inner))),
        crate::parser::Type::Void => MetalType::Int, // Void not used in our first version
    }
}
