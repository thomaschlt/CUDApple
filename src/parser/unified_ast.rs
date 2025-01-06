use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub host_code: HostCode,
    pub device_code: Vec<KernelFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostCode {
    pub statements: Vec<HostStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    VariableDecl(Declaration),
    Assign(Assignment),
    IfStmt { condition: Expression, body: Block },
}

#[derive(Debug, Clone, PartialEq)]
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
    Declaration(Declaration),
    Assignment(Assignment),
    DeviceSynchronize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceStatement {
    Declaration(Declaration),
    Assignment(Assignment),
    If(Expression, Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub var_type: Type,
    pub name: String,
    pub initializer: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Variable(String),
    IntegerLiteral(i64),
    SizeOf(Type),
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Void,
    Int,
    Float,
    SizeT,
    Dim3,
    Pointer(Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::SizeT => write!(f, "size_t"),
            Type::Dim3 => write!(f, "dim3"),
            Type::Pointer(inner) => write!(f, "{}*", inner),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    LessThan,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
}

#[derive(Debug)]
pub enum ParserError {
    HostCodeError(String),
    DeviceCodeError(String),
    BoundaryError(String),
}

impl std::error::Error for ParserError {}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::HostCodeError(msg) => write!(f, "Host code error: {}", msg),
            ParserError::DeviceCodeError(msg) => write!(f, "Device code error: {}", msg),
            ParserError::BoundaryError(msg) => write!(f, "Program boundary error: {}", msg),
        }
    }
}
