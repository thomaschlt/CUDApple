use super::ast::{KernelFunction, Type};

#[derive(Debug)]
pub enum HostStatement {
    MemoryAllocation {
        variable: String,
        size: Expression,
    },
    MemoryCopy {
        dst: String,
        src: String,
        size: Expression,
        direction: MemcpyKind,
    },
    KernelLaunch {
        kernel: String,
        grid_dim: (Expression, Expression, Expression),
        block_dim: (Expression, Expression, Expression),
        arguments: Vec<Expression>,
    },
    MemoryFree {
        variable: String,
    },
    VariableDeclaration {
        var_type: Type,
        name: String,
    },
    Assignment {
        variable: String,
        value: Expression,
    },
    DeviceSynchronize,
}

#[derive(Debug)]
pub enum MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug, PartialEq)]
pub enum Expression {
    Variable(String),
    IntegerLiteral(i64),
    SizeOf(Type),
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
}

#[derive(Debug, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Debug)]
pub struct HostProgram {
    pub statements: Vec<HostStatement>,
    pub kernels: Vec<KernelFunction>,
}
