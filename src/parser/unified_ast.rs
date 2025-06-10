use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub device_code: Vec<KernelFunction>,
    pub device_functions: Vec<DeviceFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionDefinition {
    Kernel(KernelFunction),
    Device(DeviceFunction),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    VariableDecl(Declaration),
    Assign(Assignment),
    IfStmt {
        condition: Expression,
        body: Block,
    },
    ForLoop {
        init: Box<Statement>,
        condition: Expression,
        increment: Box<Statement>,
        body: Block,
    },
    CompoundAssign {
        target: Expression,
        operator: Operator,
        value: Expression,
    },
    Return(Option<Box<Expression>>),
    Expression(Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub var_type: Type,
    pub name: String,
    pub initializer: Option<Expression>,
    pub is_shared: bool,
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
    FloatLiteral(f32),
    Infinity,
    NegativeInfinity,
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    MathFunction {
        name: String,
        arguments: Vec<Expression>,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
    },
    AtomicOperation {
        operation: String,
        target: Box<Expression>,
        value: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Void,
    Pointer(Box<Type>),
    Array(Box<Type>, i64),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Pointer(inner) => write!(f, "{0}*", inner),
            Type::Array(inner, size) => write!(f, "{0}[{1}]", inner, size),
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
    Modulo,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    Max,
    Min,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    AddressOf,
    Dereference,
    Negate,
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
    pub qualifier: Qualifier,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qualifier {
    Restrict,
    Shared,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceFunction {
    pub name: String,
    pub return_type: Type,
    pub parameters: Vec<Parameter>,
    pub body: Block,
}
