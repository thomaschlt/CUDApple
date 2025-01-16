use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub device_code: Vec<KernelFunction>,
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
    Int,
    Float,
    Void,
    Pointer(Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
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
    None,
}
