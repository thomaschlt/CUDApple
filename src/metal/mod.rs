use host::MetalKernelConfig;

use crate::parser::unified_ast::{
    CudaProgram, Dimension, Expression, KernelFunction, Operator, Statement, Type, UnaryOperator,
};
use std::fmt::Write;
pub mod host;

#[derive(Debug)]
pub struct MetalShader {
    source: String,
    config: MetalKernelConfig,
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            config: MetalKernelConfig {
                dimensions: 1,           // Default to 1D
                grid_size: (4096, 1, 1), // <-- This will be updated dynamically
                threadgroup_size: (256, 1, 1),
            },
        }
    }

    pub fn set_config(&mut self, config: MetalKernelConfig) {
        self.config = config;
    }

    pub fn generate(&mut self, program: &CudaProgram) -> Result<(), String> {
        writeln!(self.source, "            #include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            #include <metal_math>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            using namespace metal;\n").map_err(|e| e.to_string())?;

        for device_func in &program.device_functions {
            self.generate_device_function_with_indent(device_func, "            ")?;
        }

        for kernel in &program.device_code {
            self.generate_kernel_with_indent(kernel, "            ")?;
        }
        Ok(())
    }

    fn generate_device_function_with_indent(
        &mut self,
        device_func: &crate::parser::unified_ast::DeviceFunction,
        indent: &str,
    ) -> Result<(), String> {
        write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
        let return_type_str = self.translate_type(&device_func.return_type);
        writeln!(self.source, "{} {}(", return_type_str, device_func.name)
            .map_err(|e| e.to_string())?;

        let param_indent = format!("{}    ", indent);
        let mut first = true;
        for param in &device_func.parameters {
            if !first {
                writeln!(self.source, ",").map_err(|e| e.to_string())?;
            }
            first = false;

            write!(self.source, "{}", param_indent).map_err(|e| e.to_string())?;
            match &param.param_type {
                Type::Pointer(inner) => {
                    write!(
                        self.source,
                        "device {}* {}",
                        self.translate_type(inner),
                        param.name
                    )
                    .map_err(|e| e.to_string())?;
                }
                _ => {
                    write!(
                        self.source,
                        "{} {}",
                        self.translate_type(&param.param_type),
                        param.name
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
        }

        writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

        for stmt in &device_func.body.statements {
            self.translate_statement_with_indent_simple(stmt, &format!("{}    ", indent))?;
        }

        writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

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
                    // Rename 'kernel' parameter to avoid Metal keyword conflict
                    let param_name = if param.name == "kernel" {
                        "weight".to_string()
                    } else {
                        param.name.clone()
                    };

                    // Use atomic types for gradient buffers (except for SGD optimizer)
                    let is_sgd = kernel.name.contains("sgd_optimizer");
                    if param.name.starts_with("grad_") && !is_sgd {
                        write!(
                            self.source,
                            "device atomic<float>* {} [[buffer({})]]",
                            param_name, index
                        )
                        .map_err(|e| e.to_string())?;
                    } else {
                        write!(
                            self.source,
                            "device float* {} [[buffer({})]]",
                            param_name, index
                        )
                        .map_err(|e| e.to_string())?;
                    }
                }
                Type::Int => {
                    write!(
                        self.source,
                        "constant uint& {} [[buffer({})]]",
                        param.name, index
                    )
                    .map_err(|e| e.to_string())?;
                }
                Type::Float => {
                    write!(
                        self.source,
                        "constant float& {} [[buffer({})]]",
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
                    "{}uint2 thread_position_in_grid [[thread_position_in_grid]]",
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

        writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

        // Add thread index variable declarations based on dimensions
        if self.config.dimensions == 1 {
            // Check if this is linear_backward_bias (special case)
            if kernel.name.contains("linear_backward_bias") {
                // linear_backward_bias uses out_idx for 1D indexing
                writeln!(
                    self.source,
                    "{}    uint32_t out_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            } else {
                // For other 1D kernels, use batch_idx
                writeln!(
                    self.source,
                    "{}    uint32_t batch_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            }
        } else if self.config.dimensions == 2 {
            // Check kernel type for proper thread indexing
            if kernel.name.contains("linear_backward_input") {
                // linear_backward_input uses batch_idx (y) and in_idx (x)
                writeln!(
                    self.source,
                    "{}    int32_t batch_idx = thread_position_in_grid.y;",
                    indent
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    self.source,
                    "{}    int32_t in_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            } else if kernel.name.contains("linear_backward_weights") {
                // linear_backward_weights uses in_idx (y) and out_idx (x)
                writeln!(
                    self.source,
                    "{}    int32_t in_idx = thread_position_in_grid.y;",
                    indent
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    self.source,
                    "{}    int32_t out_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            } else if kernel.name.contains("linear_backward_bias") {
                // linear_backward_bias uses only out_idx (x)
                writeln!(
                    self.source,
                    "{}    int32_t out_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    self.source,
                    "{}    int32_t batch_idx = thread_position_in_grid.y;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            } else if kernel.parameters.iter().any(|p| p.name == "batch_size") {
                // Other linear kernels (forward pass)
                writeln!(
                    self.source,
                    "{}    int32_t batch_idx = thread_position_in_grid.y;",
                    indent
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    self.source,
                    "{}    int32_t out_idx = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            } else {
                // Conv2D and other kernels use out_row and out_col
                writeln!(
                    self.source,
                    "{}    int32_t out_row = thread_position_in_grid.y;",
                    indent
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    self.source,
                    "{}    int32_t out_col = thread_position_in_grid.x;",
                    indent
                )
                .map_err(|e| e.to_string())?;
            }
        }

        // Translate kernel body statements with proper indentation
        for stmt in &kernel.body.statements {
            self.translate_statement_with_indent(stmt, &format!("{}    ", indent), kernel)?;
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
        kernel: &KernelFunction,
    ) -> Result<(), String> {
        match stmt {
            Statement::VariableDecl(decl) => {
                // Check if this is a thread index calculation pattern that we can optimize
                if let Some(init) = &decl.initializer {
                    if let Expression::BinaryOp(lhs, Operator::Add, rhs) = init {
                        if let Expression::BinaryOp(inner_lhs, Operator::Multiply, inner_rhs) =
                            lhs.as_ref()
                        {
                            if is_thread_index_component(inner_lhs)
                                && is_thread_index_component(inner_rhs)
                                && is_thread_index_component(rhs)
                            {
                                // This is the thread index pattern, skip the variable declaration
                                // We'll use the Metal thread index directly in later references
                                return Ok(());
                            }
                        }
                    }
                }

                // Normal variable declaration
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                match &decl.var_type {
                    Type::Array(inner_type, size) => {
                        // Handle array declarations: float arr[10];
                        write!(
                            self.source,
                            "{} {}[{}]",
                            self.translate_type(inner_type),
                            decl.name,
                            size
                        )
                        .map_err(|e| e.to_string())?;
                        if let Some(init) = &decl.initializer {
                            write!(self.source, " = ").map_err(|e| e.to_string())?;
                            self.translate_expression(init, kernel)?;
                        }
                    }
                    Type::Int => {
                        write!(self.source, "int32_t {} = ", decl.name)
                            .map_err(|e| e.to_string())?;
                        if let Some(init) = &decl.initializer {
                            self.translate_expression(init, kernel)?;
                        }
                    }
                    _ => {
                        write!(
                            self.source,
                            "{} {} = ",
                            self.translate_type(&decl.var_type),
                            decl.name
                        )
                        .map_err(|e| e.to_string())?;
                        if let Some(init) = &decl.initializer {
                            self.translate_expression(init, kernel)?;
                        }
                    }
                }
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            Statement::IfStmt { condition, body } => {
                write!(self.source, "{}if (", indent).map_err(|e| e.to_string())?;
                self.translate_expression(condition, kernel)?;
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

                // Translate if body with additional indentation
                for stmt in &body.statements {
                    self.translate_statement_with_indent(stmt, &format!("{}    ", indent), kernel)?;
                }

                writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
            }
            Statement::Assign(assign) => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;

                // Use the improved detection function
                if is_gradient_buffer_parameter(&assign.target, kernel) {
                    // Use atomic_store_explicit for atomic parameter assignments
                    write!(self.source, "atomic_store_explicit(&").map_err(|e| e.to_string())?;
                    self.translate_expression(&assign.target, kernel)?;
                    write!(self.source, ", ").map_err(|e| e.to_string())?;
                    self.translate_expression(&assign.value, kernel)?;
                    write!(self.source, ", memory_order_relaxed)").map_err(|e| e.to_string())?;
                } else {
                    // Regular assignment for local variables and regular parameters
                    self.translate_expression(&assign.target, kernel)?;
                    write!(self.source, " = ").map_err(|e| e.to_string())?;
                    self.translate_expression(&assign.value, kernel)?;
                }
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
                            self.translate_expression(init_val, kernel)?;
                        }
                    }
                    _ => return Err("Unsupported for loop initialization".to_string()),
                }

                // Translate condition and increment
                write!(self.source, "; ").map_err(|e| e.to_string())?;
                self.translate_expression(&condition, kernel)?;
                write!(self.source, "; ").map_err(|e| e.to_string())?;

                // Handle increment
                match &**increment {
                    Statement::Assign(assign) => {
                        self.translate_expression(&assign.target, kernel)?;
                        write!(self.source, " = ").map_err(|e| e.to_string())?;
                        self.translate_expression(&assign.value, kernel)?;
                    }
                    Statement::CompoundAssign {
                        target,
                        operator,
                        value,
                    } => {
                        self.translate_expression(target, kernel)?;
                        match operator {
                            Operator::Add => write!(self.source, " += "),
                            Operator::Subtract => write!(self.source, " -= "),
                            Operator::Multiply => write!(self.source, " *= "),
                            Operator::Divide => write!(self.source, " /= "),
                            Operator::Modulo => write!(self.source, " %= "),
                            _ => return Err("Unsupported compound operator".to_string()),
                        }
                        .map_err(|e| e.to_string())?;
                        self.translate_expression(value, kernel)?;
                        writeln!(self.source, ";").map_err(|e| e.to_string())?;
                    }
                    _ => return Err("Unsupported for loop increment".to_string()),
                }

                // Close for loop header and translate body
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

                // Translate body with additional indentation
                for stmt in &body.statements {
                    self.translate_statement_with_indent(stmt, &format!("{}    ", indent), kernel)?;
                }

                writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
            }
            Statement::CompoundAssign {
                target,
                operator,
                value,
            } => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                self.translate_expression(target, kernel)?;
                match operator {
                    Operator::Add => write!(self.source, " += "),
                    Operator::Subtract => write!(self.source, " -= "),
                    Operator::Multiply => write!(self.source, " *= "),
                    Operator::Divide => write!(self.source, " /= "),
                    Operator::Modulo => write!(self.source, " %= "),
                    _ => return Err("Unsupported compound operator".to_string()),
                }
                .map_err(|e| e.to_string())?;
                self.translate_expression(value, kernel)?;
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            Statement::Return(expr_opt) => {
                write!(self.source, "{}return", indent).map_err(|e| e.to_string())?;
                if let Some(expr) = expr_opt {
                    write!(self.source, " ").map_err(|e| e.to_string())?;
                    self.translate_expression(expr, kernel)?;
                }
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            Statement::Expression(expr) => {
                // Handle atomic operations as statements
                if let Expression::AtomicOperation {
                    operation,
                    target,
                    value,
                } = expr.as_ref()
                {
                    write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                    match operation.as_str() {
                        "atomicAdd" => {
                            write!(self.source, "atomic_fetch_add_explicit(")
                                .map_err(|e| e.to_string())?;
                            // Handle the address-of operation manually for atomic operations
                            match target.as_ref() {
                                Expression::UnaryOp(UnaryOperator::AddressOf, inner_expr) => {
                                    write!(self.source, "&").map_err(|e| e.to_string())?;
                                    self.translate_expression(inner_expr, kernel)?;
                                }
                                _ => {
                                    write!(self.source, "&").map_err(|e| e.to_string())?;
                                    self.translate_expression(target, kernel)?;
                                }
                            }
                            write!(self.source, ", ").map_err(|e| e.to_string())?;
                            self.translate_expression(value, kernel)?;
                            write!(self.source, ", memory_order_relaxed)")
                                .map_err(|e| e.to_string())?;
                        }
                        _ => return Err(format!("Unsupported atomic operation: {}", operation)),
                    }
                    writeln!(self.source, ";").map_err(|e| e.to_string())?;
                } else {
                    // Handle other expressions as statements
                    write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                    self.translate_expression(expr, kernel)?;
                    writeln!(self.source, ";").map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }

    fn translate_statement_with_indent_simple(
        &mut self,
        stmt: &Statement,
        indent: &str,
    ) -> Result<(), String> {
        // Create a dummy kernel for compatibility - device functions don't use gradient buffers
        let dummy_kernel = KernelFunction {
            name: String::new(),
            parameters: Vec::new(),
            body: crate::parser::unified_ast::Block {
                statements: Vec::new(),
            },
        };
        self.translate_statement_with_indent(stmt, indent, &dummy_kernel)
    }

    fn translate_expression(
        &mut self,
        expr: &Expression,
        kernel: &KernelFunction,
    ) -> Result<(), String> {
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
                self.translate_expression(array, kernel)?;
                write!(self.source, "[").map_err(|e| e.to_string())?;
                self.translate_expression(index, kernel)?;
                write!(self.source, "]").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::BinaryOp(lhs, op, rhs) => {
                // Handle atomic parameter operations (not local variables)
                if matches!(
                    op,
                    Operator::Multiply | Operator::Add | Operator::Subtract | Operator::Divide
                ) {
                    // Skip atomic operations for SGD optimizer - it just reads gradients
                    let is_sgd = kernel.name.contains("sgd_optimizer");

                    if !is_sgd {
                        // Check if left side is an atomic gradient parameter
                        if is_gradient_buffer_parameter(lhs, kernel) {
                            write!(self.source, "atomic_load_explicit(&")
                                .map_err(|e| e.to_string())?;
                            self.translate_expression(lhs, kernel)?;
                            write!(self.source, ", memory_order_relaxed)")
                                .map_err(|e| e.to_string())?;
                            write!(self.source, " {} ", Self::operator_to_string(op))
                                .map_err(|e| e.to_string())?;
                            self.translate_expression(rhs, kernel)?;
                            return Ok(());
                        }

                        // Check if right side is an atomic gradient parameter
                        if is_gradient_buffer_parameter(rhs, kernel) {
                            self.translate_expression(lhs, kernel)?;
                            write!(self.source, " {} ", Self::operator_to_string(op))
                                .map_err(|e| e.to_string())?;
                            write!(self.source, "atomic_load_explicit(&")
                                .map_err(|e| e.to_string())?;
                            self.translate_expression(rhs, kernel)?;
                            write!(self.source, ", memory_order_relaxed)")
                                .map_err(|e| e.to_string())?;
                            return Ok(());
                        }
                    }
                }

                // Handle thread index pattern first
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
                                write!(self.source, "batch_idx").map_err(|e| e.to_string())?;
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

                // Check if we need parentheses for correct precedence
                let needs_parens = match (&**lhs, op, &**rhs) {
                    // Add parentheses for mixed precedence operations
                    (
                        Expression::BinaryOp(_, inner_op, _),
                        Operator::Divide | Operator::Multiply,
                        _,
                    ) => {
                        matches!(inner_op, Operator::Add | Operator::Subtract)
                    }
                    (
                        _,
                        Operator::Divide | Operator::Multiply,
                        Expression::BinaryOp(_, inner_op, _),
                    ) => {
                        matches!(inner_op, Operator::Add | Operator::Subtract)
                    }
                    _ => false,
                };

                if needs_parens && matches!(lhs.as_ref(), Expression::BinaryOp(_, _, _)) {
                    write!(self.source, "(").map_err(|e| e.to_string())?;
                }

                self.translate_expression(lhs, kernel)?;

                if needs_parens && matches!(lhs.as_ref(), Expression::BinaryOp(_, _, _)) {
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                }

                write!(self.source, " {} ", Self::operator_to_string(op))
                    .map_err(|e| e.to_string())?;

                if needs_parens && matches!(rhs.as_ref(), Expression::BinaryOp(_, _, _)) {
                    write!(self.source, "(").map_err(|e| e.to_string())?;
                }

                self.translate_expression(rhs, kernel)?;

                if needs_parens && matches!(rhs.as_ref(), Expression::BinaryOp(_, _, _)) {
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                }

                return Ok(());
            }
            Expression::Variable(name) => {
                if name == "NULL" {
                    write!(self.source, "0").map_err(|e| e.to_string())?;
                    return Ok(());
                }

                // Check if this is a reference to the thread index variable
                if name == "row" || name == "col" {
                    match self.config.dimensions {
                        1 => write!(self.source, "batch_idx").map_err(|e| e.to_string())?,
                        2 => {
                            if name == "col" {
                                write!(self.source, "thread_position_in_grid.x")
                                    .map_err(|e| e.to_string())?;
                            } else if name == "row" {
                                write!(self.source, "thread_position_in_grid.y")
                                    .map_err(|e| e.to_string())?;
                            } else {
                                write!(self.source, "batch_idx").map_err(|e| e.to_string())?;
                            }
                        }
                        _ => return Err("Unsupported dimensions".to_string()),
                    }
                // Add specific mapping for h and w in 2D kernels
                } else if (name == "h" || name == "w") && self.config.dimensions == 2 {
                    if name == "h" {
                        write!(self.source, "thread_position_in_grid.y")
                            .map_err(|e| e.to_string())?;
                    } else if name == "w" {
                        write!(self.source, "thread_position_in_grid.x")
                            .map_err(|e| e.to_string())?;
                    }
                // Add specific mapping for kh and kw in 2D kernels (for conv kernels)
                } else if (name == "kh" || name == "kw") && self.config.dimensions == 2 {
                    if name == "kh" {
                        write!(self.source, "thread_position_in_grid.y")
                            .map_err(|e| e.to_string())?;
                    } else if name == "kw" {
                        write!(self.source, "thread_position_in_grid.x")
                            .map_err(|e| e.to_string())?;
                    }
                // Add specific mapping for c_out in 1D kernels (for conv bias backward)
                } else if name == "c_out" && self.config.dimensions == 1 {
                    write!(self.source, "thread_position_in_grid.x").map_err(|e| e.to_string())?;
                // Add mapping for common thread index variable names
                } else if name == "idx" || name == "tid" {
                    match self.config.dimensions {
                        1 => write!(self.source, "batch_idx").map_err(|e| e.to_string())?,
                        2 => write!(self.source, "thread_position_in_grid.x")
                            .map_err(|e| e.to_string())?,
                        _ => return Err("Unsupported dimensions".to_string()),
                    }
                } else if name == "i" {
                    // Smart handling: only map 'i' to batch_idx if it's likely a thread index
                    let source_so_far = &self.source;
                    if source_so_far.lines().last().unwrap_or("").contains("for (") {
                        // We're in a for loop declaration, keep 'i' as is
                        write!(self.source, "i").map_err(|e| e.to_string())?;
                    } else {
                        // Likely a thread index, map to batch_idx
                        match self.config.dimensions {
                            1 => write!(self.source, "batch_idx").map_err(|e| e.to_string())?,
                            2 => write!(self.source, "thread_position_in_grid.x")
                                .map_err(|e| e.to_string())?,
                            _ => return Err("Unsupported dimensions".to_string()),
                        }
                    }
                } else if name == "kernel" {
                    write!(self.source, "weight").map_err(|e| e.to_string())?;
                } else if name == "sample_idx" {
                    // Map sample_idx to batch_idx for training loops
                    write!(self.source, "batch_idx").map_err(|e| e.to_string())?;
                } else {
                    write!(self.source, "{}", name).map_err(|e| e.to_string())?;
                }
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
                "max" | "fmaxf" => {
                    write!(self.source, "max(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ", ").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[1], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "min" | "fminf" => {
                    write!(self.source, "min(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ", ").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[1], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "expf" => {
                    write!(self.source, "exp(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "logf" => {
                    write!(self.source, "log(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "sqrtf" => {
                    write!(self.source, "sqrt(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "powf" => {
                    write!(self.source, "pow(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ", ").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[1], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "tanhf" => {
                    write!(self.source, "tanh(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                "fabsf" => {
                    write!(self.source, "abs(").map_err(|e| e.to_string())?;
                    self.translate_expression(&arguments[0], kernel)?;
                    write!(self.source, ")").map_err(|e| e.to_string())?;
                    return Ok(());
                }
                _ => return Err(format!("Unsupported math function: {:?}", name)),
            },
            Expression::FunctionCall { name, arguments } => {
                write!(self.source, "{}(", name).map_err(|e| e.to_string())?;
                for (i, arg) in arguments.iter().enumerate() {
                    if i > 0 {
                        write!(self.source, ", ").map_err(|e| e.to_string())?;
                    }
                    self.translate_expression(arg, kernel)?;
                }
                write!(self.source, ")").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::Infinity => {
                write!(self.source, "INFINITY").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::NegativeInfinity => {
                write!(self.source, "-INFINITY").map_err(|e| e.to_string())?;
                return Ok(());
            }
            Expression::AtomicOperation {
                operation,
                target,
                value,
            } => {
                match operation.as_str() {
                    "atomicAdd" => {
                        write!(self.source, "atomic_fetch_add_explicit(")
                            .map_err(|e| e.to_string())?;
                        // Handle the address-of operation manually for atomic operations
                        match target.as_ref() {
                            Expression::UnaryOp(UnaryOperator::AddressOf, inner_expr) => {
                                write!(self.source, "&").map_err(|e| e.to_string())?;
                                self.translate_expression(inner_expr, kernel)?;
                            }
                            _ => {
                                write!(self.source, "&").map_err(|e| e.to_string())?;
                                self.translate_expression(target, kernel)?;
                            }
                        }
                        write!(self.source, ", ").map_err(|e| e.to_string())?;
                        self.translate_expression(value, kernel)?;
                        write!(self.source, ", memory_order_relaxed)")
                            .map_err(|e| e.to_string())?;
                    }
                    _ => return Err(format!("Unsupported atomic operation: {}", operation)),
                }
                return Ok(());
            }
            Expression::UnaryOp(op, expr) => {
                match op {
                    UnaryOperator::AddressOf => write!(self.source, "&"),
                    UnaryOperator::Dereference => write!(self.source, "*"),
                    UnaryOperator::Negate => write!(self.source, "-"),
                }
                .map_err(|e| e.to_string())?;
                self.translate_expression(expr, kernel)?;
                return Ok(());
            }
        }
    }

    fn operator_to_string(op: &Operator) -> &'static str {
        match op {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Multiply => "*",
            Operator::Divide => "/",
            Operator::Modulo => "%",
            Operator::LessThan => "<",
            Operator::LessEqual => "<=",
            Operator::GreaterThan => ">",
            Operator::GreaterEqual => ">=",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
            Operator::Max => "max",
            Operator::Min => "min",
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn source_without_headers(&self) -> String {
        // Return the source without the Metal headers
        let lines: Vec<&str> = self.source.lines().collect();
        if lines.len() >= 3
            && lines[0].contains("#include <metal_stdlib>")
            && lines[1].contains("#include <metal_math>")
            && lines[2].contains("using namespace metal;")
        {
            // Skip the first 3 header lines and return the rest
            lines[3..].join("\n")
        } else {
            // If headers are not where expected, return as is
            self.source.clone()
        }
    }

    fn translate_type(&self, t: &Type) -> String {
        match t {
            Type::Int => "int32_t".to_string(),
            Type::Float => "float".to_string(),
            Type::Void => "void".to_string(),
            Type::Pointer(inner) => format!("device {}*", self.translate_type(inner)),
            Type::Array(inner, size) => format!("{}[{}]", self.translate_type(inner), size),
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

fn is_gradient_buffer_parameter(expr: &Expression, kernel: &KernelFunction) -> bool {
    match expr {
        Expression::ArrayAccess { array, .. } => {
            if let Expression::Variable(var_name) = array.as_ref() {
                // Only treat as atomic if it's a function parameter starting with "grad_"
                var_name.starts_with("grad_")
                    && kernel.parameters.iter().any(|p| p.name == *var_name)
            } else {
                false
            }
        }
        _ => false,
    }
}
